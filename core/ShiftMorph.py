# @inproceedings{yang2024shiftmorph,
#   title={ShiftMorph: A Fast and Robust Convolutional Neural Network for 3D Deformable Medical Image Registration},
#   author={Yang, Lijian and Li, Weisheng and Shu, Yucheng and Mi, Jianxun and Huang, Yuping and Xiao, Bin},
#   booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
#   pages={2814--2823},
#   year={2024}
# }

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mikuti.geometry import Morpher, to_world_flow
from typing import List

BIAS = True
Normlizer = lambda ch: nn.InstanceNorm3d(ch)
Activation = lambda: nn.PReLU()
Pool = lambda: nn.AvgPool3d(2, 2)


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=BIAS)
        self.norm = Normlizer(out_channels)
        self.activation = Activation()


class ShiftDecoder(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(skip_channels, skip_channels, 4, 2, 1, bias=BIAS),
            Normlizer(skip_channels),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels + skip_channels, out_channels, 4, 2, 1, bias=BIAS),
            Normlizer(out_channels),
            Activation(),
        )

    def forward(self, x, skip):
        x = torch.cat([x, self.down(skip)], dim=1)
        x = self.conv(x)
        return x


class ShiftEmbedding(torch.nn.Module):

    def __init__(self, in_channels=2, embed_dim=16, patch_size=4, stride=2, padding=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim, patch_size, stride, padding, bias=BIAS),
            Normlizer(embed_dim),
        )
        self.SHIFTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    def forward(self, x: torch.Tensor, n_shifts=8, flip_second=True):
        B, C, H, W, T = x.shape
        x = F.pad(x, pad=(0, 1, 0, 1, 0, 1))
        shifts = []
        shift_pos = self.SHIFTS[:n_shifts]
        for idx in range(n_shifts):
            i, j, k = shift_pos[idx]
            if not self.training or not flip_second or idx < 4:
                shifts.append(x[:, :, i : i + H, j : j + W, k : k + T])
            else:
                # flip the channel to reverse the registration direction
                shifts.append(x[:, :, i : i + H, j : j + W, k : k + T].flip(dims=[1]))

        shifts = torch.cat(shifts, dim=1).view(B * n_shifts, C, H, W, T)
        embed = self.proj(shifts)
        return embed


class ShiftMorphBase(nn.Module):

    def __init__(self, embed_dim, scale=0.3) -> None:
        super().__init__()
        CONV = ConvBlock
        ch = embed_dim
        self.range_flow = scale
        self.embedding = ShiftEmbedding(2, ch, 4, 2, 1)
        self.conv1 = CONV(ch, ch * 2)
        self.down1 = Pool()
        self.conv2 = CONV(ch * 2, ch * 4)
        self.down2 = Pool()
        self.conv3 = CONV(ch * 4, ch * 8)
        self.down3 = Pool()
        self.bottle = CONV(ch * 8, ch * 16)
        self.decoder3 = ShiftDecoder(ch * 16, ch * 8, ch * 8)
        self.decoder2 = ShiftDecoder(ch * 8, ch * 4, ch * 4)
        self.decoder1 = ShiftDecoder(ch * 4, ch * 2, ch * 2)

        self.reg_head = nn.Sequential(
            ConvBlock(ch * 2, ch),
            ConvBlock(ch, ch),
            nn.Conv3d(ch, 3, 3, 1, 1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2.0, mode="trilinear", align_corners=False),
        )
        self.reg_head[2].weight.data.normal_(0, 1e-3)
        self.reg_head[2].bias.data.zero_()

    def forward(self, fix: torch.Tensor, mov: torch.Tensor, n_shifts=8, groups=2, flip=True) -> torch.Tensor:
        x_in = torch.cat([fix, mov], dim=1)
        B, C, H, W, D = x_in.shape
        embedded = self.embedding(x_in, n_shifts, flip)

        f1 = self.conv1(embedded)
        f2 = self.conv2(self.down1(f1))
        f3 = self.conv3(self.down2(f2))
        out = self.bottle(self.down3(f3))

        # shift merging
        f1 = f1.view(B * groups, -1, *f1.shape[1:]).mean(dim=1, keepdim=False)
        f2 = f2.view(B * groups, -1, *f2.shape[1:]).mean(dim=1, keepdim=False)
        f3 = f3.view(B * groups, -1, *f3.shape[1:]).mean(dim=1, keepdim=False)
        out = out.view(B * groups, -1, *out.shape[1:]).mean(dim=1, keepdim=False)

        out = self.decoder3(out, f3)
        out = self.decoder2(out, f2)
        out = self.decoder1(out, f1)

        flow = self.reg_head(out) * self.range_flow
        flow = to_world_flow(flow, [H, W, D], dim=1, align_corners=True)
        flow = flow.view(B, -1, *flow.shape[2:])

        if groups == 2:
            flow, inv_flow = torch.split(flow, split_size_or_sections=3, dim=1)
            inv_flow = inv_flow if flip else inv_flow.negative()
            return flow, inv_flow
        else:
            return flow


class ShiftMorph(nn.Module):

    def __init__(self, num_iter, embed_dim, diff=0, scale=0.3, gamma=0.01, flip=True, share=True) -> None:
        super().__init__()
        self.num_iter = num_iter
        self.diff = diff
        self.scale = scale
        self.gamma = gamma
        self.flip = flip
        self.share = share
        self.stn = Morpher()
        self.extra_loss = None

        if self.share:
            self.nets: List[ShiftMorphBase] = nn.ModuleList([ShiftMorphBase(embed_dim, scale)])
        else:
            self.nets: List[ShiftMorphBase] = nn.ModuleList([ShiftMorphBase(embed_dim, scale) for _ in range(num_iter)])

    def forward(self, fix: torch.Tensor, mov: torch.Tensor) -> torch.Tensor:
        if self.training:
            flow, inv_flow = self.full_forward(fix, mov)
            zero_flow = self.stn.compose(flow, inv_flow)
            self.extra_loss = {"consist": zero_flow.pow(2).mean() * self.gamma}
        else:
            flow = self.half_forward(fix, mov)
        return flow

    def full_forward(self, fix, mov):
        num_net = len(self.nets)

        flow, inv_flow = self.nets[0].forward(fix, mov, 8, 2, self.flip)
        composed_flow = self.stn.diffeomorphic(flow, self.diff)
        composed_inv_flow = self.stn.diffeomorphic(inv_flow, self.diff)

        for idx in range(1, self.num_iter):
            warped = self.stn.warp(mov, composed_flow)

            flow, inv_flow = self.nets[idx % num_net].forward(fix, warped, 8, 2, self.flip)
            flow = self.stn.diffeomorphic(flow, self.diff)
            inv_flow = self.stn.diffeomorphic(inv_flow, self.diff)

            composed_flow = self.stn.compose(composed_flow, flow)
            composed_inv_flow = self.stn.compose(composed_inv_flow, inv_flow)
        return composed_flow, composed_inv_flow

    def half_forward(self, fix, mov):
        num_net = len(self.nets)

        flow = self.nets[0].forward(fix, mov, 4, 1, self.flip)
        composed_flow = self.stn.diffeomorphic(flow, self.diff)

        for idx in range(1, self.num_iter):
            warped = self.stn.warp(mov, composed_flow)

            flow = self.nets[idx % num_net].forward(fix, warped, 4, 1, self.flip)
            flow = self.stn.diffeomorphic(flow, self.diff)

            composed_flow = self.stn.compose(composed_flow, flow)
        return composed_flow
