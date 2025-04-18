import torch
import torch.amp
import torch.nn.functional as F
import math
from typing import List, Tuple
import numpy as np


def to_world_points(kpts: torch.Tensor, spatial_size: List[int], dim: int = -1, align_corners: bool = True) -> torch.Tensor:
    device = kpts.device
    view_shape: List[int] = kpts.ndim * [1]
    view_shape[dim] = -1
    size_tensor = torch.tensor(spatial_size, device=device).reshape(view_shape)

    kpts = kpts.flip(dim)  # to world dimension order
    if not align_corners:
        kpts *= size_tensor / (size_tensor - 1)

    kpts = (kpts + 1) * 0.5 * (size_tensor - 1)
    return kpts


def to_torch_points(kpts: torch.Tensor, spatial_size: List[int], dim: int = -1, align_corners: bool = True) -> torch.Tensor:

    device = kpts.device
    view_shape: List[int] = kpts.ndim * [1]
    view_shape[dim] = -1
    size_tensor = torch.tensor(spatial_size, device=device).reshape(view_shape)

    kpts = kpts / (size_tensor - 1) * 2 - 1
    # align_corners: [-1, 1] i.e. [0, H-1] places H pixels, space length = H
    # align_corners:              [0.5, H-1.5] places H pixels, space length = H - 1
    # NAC (not align corners)
    # AC (align corners)
    # thus: NAC = AC * (H-1) / H
    if not align_corners:
        kpts *= (size_tensor - 1) / size_tensor

    kpts = kpts.flip(dim)  # To pytorch dimension order
    return kpts


def to_world_flow(flow: torch.Tensor, spatial_size: List[int], dim: int = -1, align_corners: bool = True) -> torch.Tensor:
    device = flow.device
    view_shape: List[int] = flow.ndim * [1]
    view_shape[dim] = -1
    size_tensor = torch.tensor(spatial_size, device=device).reshape(view_shape)

    flow = flow.flip(dim)  # to world dimension order
    if not align_corners:
        flow *= size_tensor / (size_tensor - 1)

    flow *= 0.5 * (size_tensor - 1)
    return flow


def to_torch_flow(flow: torch.Tensor, spatial_size: List[int], dim: int = -1, align_corners: bool = True) -> torch.Tensor:
    device = flow.device
    view_shape: List[int] = flow.ndim * [1]
    view_shape[dim] = -1
    size_tensor = torch.tensor(spatial_size, device=device).reshape(view_shape)

    flow *= 2 / (size_tensor - 1)
    if not align_corners:
        flow *= (size_tensor - 1) / size_tensor
    flow = flow.flip(dim)  # To pytorch dimension order

    return flow


def get_identity_grid(shape: List[int], device: str = "cpu") -> torch.Tensor:
    grid = torch.meshgrid([torch.arange(0, sz, device=device) for sz in shape], indexing="ij")
    grid = torch.stack(grid, dim=-1)
    return grid


def warp_image(
    img: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
    align_corners: bool = True,
    padding_mode="zeros",
    indentity_grid: torch.Tensor = None,
) -> torch.Tensor:

    shape = img.shape[2:]
    device = str(img.device)

    grid = indentity_grid if indentity_grid is not None else get_identity_grid(shape, device=device)
    grid = grid + flow.moveaxis(1, -1)
    grid = to_torch_points(grid, shape, align_corners=align_corners)

    warped_img = F.grid_sample(img, grid, mode=mode, align_corners=align_corners, padding_mode=padding_mode)
    return warped_img


def resize_flow(flow: torch.Tensor, size: List[int] = None, scale_factor: float = None, mode: str = None, align_corners: bool = True) -> torch.Tensor:

    pre_size = torch.tensor(flow.shape[2:])
    ndim = len(pre_size)

    if size is not None:
        size = torch.tensor(size)
    else:
        size = pre_size.mul(scale_factor).long()

    ratio = size / pre_size
    size = size.tolist()
    if ndim == 2:
        ratio = ratio.view(1, 2, 1, 1)
        mode = mode or "bilinear"
    else:
        ratio = ratio.view(1, 3, 1, 1, 1)
        mode = mode or "trilinear"

    resized: torch.Tensor = ratio.type_as(flow) * F.interpolate(flow, size=size, mode=mode, align_corners=align_corners)
    return resized


class Morpher(torch.nn.Module):
    grid: torch.Tensor
    shape: torch.Size

    def __init__(self, align_corners=True, padding_mode="zeros") -> None:
        super().__init__()
        self.shape = None
        self.params = dict(
            align_corners=align_corners,
            padding_mode=padding_mode,
        )

    def identity_grid_cache(self, shape, device):
        if self.shape != shape:
            grid = get_identity_grid(shape, str(device))
            self.shape = shape
            self.register_buffer("grid", grid, persistent=False)

        if self.grid.device != device:
            self.grid = self.grid.to(device)

        return self.grid

    def warp(self, src: torch.Tensor, flow: torch.Tensor, mode="bilinear"):

        grid = self.identity_grid_cache(flow.shape[2:], flow.device)
        warped = warp_image(src, flow, mode=mode, indentity_grid=grid, **self.params)

        return warped

    def diffeomorphic(self, flow: torch.Tensor, inter_steps=7, time_shift=0, mode="bilinear"):
        if inter_steps + time_shift == 0:
            return flow

        flow = flow / (1 << inter_steps)
        for _ in range(inter_steps + time_shift):
            flow = flow + self.warp(flow, flow, mode)

        return flow

    def compose(self, flow1, flow2, mode="bilinear"):
        flow = flow2 + self.warp(flow1, flow2, mode)
        return flow


def mindssc(img, dilate=2, radius=2):
    device = img.device

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]], dtype=torch.float, device=device)

    def pdist(x):
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
        return dist

    # squared distances
    dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6, device=device), torch.arange(6, device=device), indexing="ij")
    mask = (x > y).view(-1) & (dist == 2).view(-1)

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :].long()
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :].long()
    mshift1 = torch.zeros((12, 1, 3, 3, 3), device=device)
    mshift1.view(-1)[torch.arange(12, device=device) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros((12, 1, 3, 3, 3), device=device)
    mshift2.view(-1)[torch.arange(12, device=device) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad = torch.nn.ReplicationPad3d(dilate)

    # compute patch-ssd
    ssd = (F.conv3d(rpad(img), mshift1, dilation=dilate) - F.conv3d(rpad(img), mshift2, dilation=dilate)) ** 2

    K = radius * 2 + 1
    ssd = F.avg_pool3d(ssd, K, 1, K // 2)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind)

    return mind
