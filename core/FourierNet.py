# @inproceedings{jia2023fourier,
#   title={Fourier-net: Fast image registration with band-limited deformation},
#   author={Jia, Xi and Bartlett, Joseph and Chen, Wei and Song, Siyang and Zhang, Tianyang and Cheng, Xinxing and Lu, Wenqi and Qiu, Zhaowen and Duan, Jinming},
#   booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
#   volume={37},
#   number={1},
#   pages={1015--1023},
#   year={2023}
# }

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BIAS = True
SCALE = 0.3
Normlizer = lambda ch: nn.InstanceNorm3d(ch)
Activation = lambda: nn.PReLU()
Pool = lambda in_ch, out_ch: nn.Conv3d(in_ch, out_ch, 2, 2, 0)


class FourierNet(nn.Module):
    def __init__(self, start_channel=16, **args):
        super(FourierNet, self).__init__()
        self.in_channel = 2
        self.n_classes = 3
        self.start_channel = start_channel

        bias_opt = BIAS

        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.r_dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.rr_dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.r_up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.r_up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            Normlizer(out_channels),
            nn.PReLU(),
        )
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias),
            Normlizer(out_channels),
            nn.PReLU(),
        )
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
        layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        return layer

    def forward(self, fix: torch.Tensor, mov: torch.Tensor) -> torch.Tensor:
        x_in = torch.cat([fix, mov], dim=1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        r_d0 = torch.cat((self.r_up1(e4), e3), 1)

        r_d0 = self.r_dc1(r_d0)
        r_d0 = self.r_dc2(r_d0)

        r_d1 = torch.cat((self.r_up2(r_d0), e2), 1)

        r_d1 = self.r_dc3(r_d1)
        r_d1 = self.r_dc4(r_d1)

        flow = self.rr_dc9(r_d1)
        flow = torch.fft.fftshift(torch.fft.fftn(flow, dim=[-3, -2, -1]), dim=[-3, -2, -1])

        diff_shape = np.array(x_in.shape[-3:]) - np.array(flow.shape[-3:])
        pad_l = diff_shape // 2
        pad_r = diff_shape - pad_l
        HL, WL, TL = pad_l
        HR, WR, TR = pad_r

        flow = F.pad(flow, [TL, TR, WL, WR, HL, HR], mode="constant", value=0.0)
        flow = torch.fft.ifftn(torch.fft.ifftshift(flow, dim=[-3, -2, -1]), dim=[-3, -2, -1])
        flow = torch.real(flow)

        return flow
