import torch
import torch.nn.functional as F
import numpy as np


def tre_loss(fixed, moving, voxel_size=None, eps=1e-6) -> torch.Tensor:
    """
    Computes target registration error (TRE) loss for keypoint matching.
    vx: the length of one voxel
    """
    if voxel_size is None:
        return ((fixed - moving) ** 2 + eps).sum(-1).sqrt().nanmean()
    else:
        return ((fixed - moving).mul(voxel_size) ** 2 + eps).sum(-1).sqrt().nanmean()


def PSNR(x: torch.Tensor, y: torch.Tensor, reduce=True):
    mse = torch.pow((x - y), 2).mean(dim=[-1, -2, -3])
    ret = -10 * torch.log10(mse)

    if reduce:
        ret = ret.mean()
    return ret


def SNR(S: torch.Tensor, N: torch.Tensor, reduce=True):
    S = S.flatten(1)
    N = N.flatten(1)
    res = torch.norm(S, p=2, dim=-1)
    denom = torch.norm(N, p=2, dim=-1)
    ret = 20 * torch.log10(res / (denom + 1e-30))

    if reduce:
        ret = ret.mean()

    return ret


def IoU_loss(y_pred: torch.Tensor, y_true: torch.Tensor, reduce=True, eps=1e-5):
    y_pred = torch.flatten(y_pred, start_dim=2)
    y_true = torch.flatten(y_true, start_dim=2)

    intersection = (y_pred * y_true).sum(dim=-1)
    union = (y_pred + y_true).sum(dim=-1) - intersection
    iou = (intersection + eps) / (union + eps)
    if reduce:
        iou = torch.mean(iou)
    return -iou


def dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor, reduce=True, eps=1e-6):
    y_pred = torch.flatten(y_pred, start_dim=2)
    y_true = torch.flatten(y_true, start_dim=2)

    intersection = (y_pred * y_true).sum(dim=-1)
    union = (y_pred + y_true).sum(dim=-1)
    dsc = (2.0 * intersection) / (union + eps)
    if reduce:
        dsc = torch.mean(dsc)
    return -dsc


def lncc_loss(x: torch.Tensor, y: torch.Tensor, kernel_size=9, reduce=True, eps=1e-9):
    ks = kernel_size
    B, C = x.shape[:2]
    ndims = len(x.shape) - 2
    avg_pool = F.avg_pool3d if ndims == 3 else F.avg_pool2d
    x = F.pad(x, pad=[ks // 2] * ndims * 2, mode="replicate")
    y = F.pad(y, pad=[ks // 2] * ndims * 2, mode="replicate")

    x_mean = avg_pool(x, ks, 1)
    y_mean = avg_pool(y, ks, 1)
    xy_mean = avg_pool(x * y, ks, 1)
    x2_mean = avg_pool(x * x, ks, 1)
    y2_mean = avg_pool(y * y, ks, 1)

    cross = xy_mean - x_mean * y_mean
    x_var = x2_mean - x_mean * x_mean
    y_var = y2_mean - y_mean * y_mean

    ncc = (cross * cross) / (x_var * y_var + eps)
    ncc = ncc.view(B, C, -1).mean(dim=-1)
    if reduce:
        ncc = torch.mean(ncc)
    return -ncc

def spatial_grad_loss(pred: torch.Tensor, reduce=True):
    ndims = len(pred.shape) - 2
    energy = 0.0

    for dim in range(ndims):
        energy += pred.diff(n=1, dim=dim + 2).pow(2).flatten(start_dim=2).mean(dim=-1)
    energy /= ndims

    if reduce:
        energy = energy.mean()
    return energy


def bending_energy_loss(pred: torch.Tensor, reduce=True):
    ndims = len(pred.shape) - 2
    energy = 0.0

    for dim_x in range(ndims):
        dxdx = pred.diff(n=2, dim=dim_x + 2).pow(2).flatten(start_dim=2).mean(dim=-1)
        energy += dxdx
        for dim_y in range(dim_x + 1, ndims):
            dxdy = pred.diff(n=1, dim=dim_x + 2).diff(n=1, dim=dim_y + 2).pow(2).flatten(start_dim=2).mean(dim=-1)
            energy += 2 * dxdy
    energy /= ndims * ndims

    if reduce:
        energy = energy.mean()
    return energy


def JacobianDet3D(flow: torch.Tensor, sample_grid=None):
    if sample_grid is None:
        J = flow
    else:
        J = flow + sample_grid

    J = torch.moveaxis(J, 1, -1).contiguous()
    dz = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dy = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    if sample_grid is None:
        dz[..., 0] += 1
        dy[..., 1] += 1
        dx[..., 2] += 1

    Jdet0 = dz[..., 0] * (dy[..., 1] * dx[..., 2] - dy[..., 2] * dx[..., 1])
    Jdet1 = dz[..., 1] * (dy[..., 0] * dx[..., 2] - dy[..., 2] * dx[..., 0])
    Jdet2 = dz[..., 2] * (dy[..., 0] * dx[..., 1] - dy[..., 1] * dx[..., 0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def JacobianDet2D(flow: torch.Tensor, sample_grid=None):
    if sample_grid is None:
        J = flow
    else:
        J = flow + sample_grid

    J = torch.moveaxis(J, 1, -1).contiguous()
    dy = J[:, 1:, :-1, :] - J[:, :-1, :-1, :]
    dx = J[:, :-1, 1:, :] - J[:, :-1, :-1, :]

    if sample_grid is None:
        dy[..., 0] += 1
        dx[..., 1] += 1

    Jdet = dy[..., 0] * dx[..., 1] - dy[..., 1] * dx[..., 0]

    return Jdet


def neg_Jacdet_percent(flow: torch.Tensor):
    if flow.ndim == 4:
        jac = JacobianDet2D(flow)
    else:
        jac = JacobianDet3D(flow)
    ratio = (jac < 0).float().mean() * 100
    return ratio


@torch.jit.script
def charbonnier_Loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    diff = x - y
    loss = torch.mean(torch.sqrt(diff * diff + eps))
    return loss
