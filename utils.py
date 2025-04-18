import numpy as np
import pickle as pkl
import torch
import random
import nibabel as nib
import os
import matplotlib.pyplot as plt
from typing import List
import shutil
import json
import math


def load_state_dict(model: torch.nn.Module, path, key="net", strict=True):
    state = torch.load(path, map_location="cpu")
    if key is not None:
        state = state[key]
    model.load_state_dict(state, strict=strict)
    model.weight_path = path
    return model


def dict_sort_as(data, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = data[key]
    return new_dict


def endless_generater(loader):
    while True:
        for data in loader:
            yield data


def batch_dict_items(data):
    for key in data:
        value = data[key]
        if isinstance(value, (np.ndarray, torch.Tensor)):
            value = torch.as_tensor(value)[None, ...]
        data[key] = value
    return data


def init_environment(seed, device=0, cudnn=True, benchmark=True, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = cudnn
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nibload(path: str) -> np.ndarray:
    data = nib.load(path).get_fdata()
    return data


def nibsave(data: np.ndarray, path: str, affine=None):
    # data :
    #   axis 0: from right to left
    #   axis 1: from top to bottom
    #   axis 2: from posterior to anterior
    if affine is None:
        affine = np.array(
            [
                [-1, 0, 0, 0],  #
                [0, 0, 1, 0],  #
                [0, -1, 0, 0],  #
                [0, 0, 0, 1],
            ]
        )
    data = np.array(data, dtype="float64")
    nifty = nib.Nifti1Image(data, affine)
    nib.save(nifty, path)


def pkload(path):
    data = None
    with open(path, "rb") as file:
        data = pkl.load(file)
    return data


def pklsave(data, path):
    with open(path, "wb") as file:
        pkl.dump(data, file)


def jsonload(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def jsonsave(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def three_view(img, indices=None, cmap="gray", ax=None, title=None, alpha=1, enhance=False):
    img = np.array(img).squeeze()

    if indices is None:
        indices = [s // 2 for s in img.shape[-3:]]
    elif isinstance(indices, int):
        indices = [indices, indices, indices]

    if ax is None:
        fig, ax = plt.subplots(1, 3)
    for i in range(3):
        tmp = np.moveaxis(img, i, 0)[indices[i], ...]
        if enhance:
            from skimage.exposure import equalize_adapthist

            tmp = tmp - tmp.min()
            tmp = tmp / (tmp.max() + 1e-8)
            tmp = equalize_adapthist(tmp, clip_limit=0.1, nbins=50)
        ax[i].imshow(tmp, cmap=cmap, alpha=alpha)
        ax[i].axis("off")

    if title is not None:
        ax[1].title(title)


def index2onehot(indice: torch.Tensor, shape) -> torch.Tensor:
    numel = np.prod(shape)
    out = torch.sparse_coo_tensor(indice, torch.ones_like(indice).flatten(), size=[numel])
    out = out.to_dense().view(*shape).float()
    return out


def onehot2value(seg: np.ndarray):
    unknow = seg.sum(axis=0, keepdims=True) == 0
    seg = np.concatenate([unknow, seg], axis=0)
    seg = np.argmax(seg, axis=0)
    return seg


def onehot(seg: torch.Tensor, num_class=None, ignore_background=True):
    B, C, H, W, T = seg.shape
    seg = seg.flatten()
    if num_class is None:
        num_class = len(torch.unique(seg))
    seg = torch.nn.functional.one_hot(seg, num_class)
    if ignore_background:
        seg = seg[:, 1:]
    seg = seg.view(B, H, W, T, -1).permute(0, 4, 1, 2, 3).contiguous().long()
    return seg


def tbadd_dict(tbwriter, data_dict: dict, index, group="loss"):
    for key, value in data_dict.items():
        tbwriter.add_scalar(f"{group}/{key}", value, index)


def format_dict(dict_data: dict, ncols=None) -> str:
    output = ""
    ncols = ncols or 4096
    cnt = 0
    for key, value in dict_data.items():
        cnt = cnt + 1
        if isinstance(value, float):
            if abs(value) > 1e30:
                value = "inf"
            elif abs(value) > 1e-3:
                value = f"{value:.4f}".rstrip("0").rstrip(".")
            elif abs(value) > 1e-30:
                left, right = f"{value:.4e}".split("e")
                value = left.rstrip("0").rstrip(".") + "e" + right
            else:
                value = "0"
        output += f"{key}={value} | "
        if cnt == ncols:
            cnt = 0
            output += "\n"

    return output.strip("\n")


def backup_files(path_list, destination):
    for path in path_list:
        target = os.path.join(destination, os.path.basename(path))
        if os.path.isdir(path):
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy(path, target, follow_symlinks=True)


def check_tensor(tmp: torch.Tensor | np.ndarray, tag=None):
    tag = f"{tag}: " if tag else ""
    if isinstance(tmp, torch.Tensor):
        meta = {
            "shape": " x ".join([str(s) for s in tmp.shape]),
            "min": tmp.min().item(),
            "max": tmp.max().item(),
            "mean": tmp.mean().item(),
            "std": tmp.std().item(),
            "count_nan": tmp.isnan().sum(),
            "device": tmp.device,
            "dtype": tmp.dtype,
        }
    else:
        meta = {
            "shape": " x ".join([str(s) for s in tmp.shape]),
            "min": tmp.min(),
            "max": tmp.max(),
            "mean": tmp.mean(),
            "std": tmp.std(),
            "count_nan": np.isnan(tmp).sum(),
            "device": "cpu",
            "dtype": tmp.dtype,
        }
    print(tag + format_dict(meta))


class Accumulator:

    def __init__(self, cache=False) -> None:
        super().__init__()
        self.cnt = 0
        self.dict_sum = {}
        self.dict_cache = {}
        self.cache = cache

    def append(self, data: dict):
        for key, value in data.items():
            if key in self.dict_sum:
                self.dict_sum[key] += value
                if self.cache:
                    self.dict_cache[key].append(value)
            else:
                self.dict_sum[key] = value
                if self.cache:
                    self.dict_cache[key] = [value]
        self.cnt += 1

    def average(self):
        data = {}
        for key, value in self.dict_sum.items():
            if not isinstance(value, str):
                value = value / self.cnt
                data[key] = value
        return data

    @torch.no_grad()
    def all_reduce_average(self):
        value_list = [torch.tensor(self.cnt)]
        key_list = []
        for key, value in self.dict_sum.items():
            value = torch.tensor(value)
            value_list.append(value)
            key_list.append(key)
        value_list = torch.stack(value_list).float().cuda()
        torch.distributed.all_reduce(value_list, torch.distributed.ReduceOp.SUM)
        value_list = value_list[1:] / value_list[0]
        data = {key: value.item() for key, value in zip(key_list, value_list)}
        return data

    def reset(self):
        self.cnt = 0
        self.dict_sum = {}


def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, weight_decay=1e-5):
    g = [], []
    norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    act = tuple(v for k, v in torch.nn.__dict__.items() if "ReLU" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if isinstance(v, norm) or isinstance(v, act):
                g[1].append(p)
            elif p_name == "bias":  # bias (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(g[0], lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[0], lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[0], lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[0], lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    return optimizer


def cosine_annealing_cycle(epoch, T0, Tmul, f_min):
    if Tmul == 1:
        Tcur = epoch % T0
        Tend = T0
    else:
        n = int(math.log((epoch / T0 * (Tmul - 1) + 1), Tmul))
        Tcur = epoch - T0 * (Tmul**n - 1) / (Tmul - 1)
        Tend = T0 * (Tmul**n)

    factor = (1 + math.cos(math.pi * Tcur / Tend)) / 2 * (1 - f_min) + f_min
    return factor
