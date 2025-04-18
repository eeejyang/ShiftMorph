import os
import time
import numpy as np
import argparse
from functools import partial
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from monai.data import DataLoader
from datasets import OASISDataset
from geometry import Morpher
import metric
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=777, help="random seed")
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
parser.add_argument("--amp", action="store_true", default=False, help="whether to use amp for fast forward")
parser.add_argument("--lmd_sim", type=float, default=1.0, help="weight for similarity loss")
parser.add_argument("--lmd_smooth", type=float, default=1.0, help="weight for smoothness loss")
parser.add_argument("--lmd_dice", type=float, default=0.0, help="weight for dice loss")  # 纯无监督就设置为0
parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
parser.add_argument("--lr", type=float, default=1e-4, help="learning_rate")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
parser.add_argument("--max_epoch", type=int, default=300, help="maximum epoch")
parser.add_argument("--T0", type=int, default=300, help="T0 for cosine annealing")
parser.add_argument("--Tmul", type=int, default=1, help="Tmul for cosine annealing")
parser.add_argument("--print_freq", type=int, default=100, help="verbose frequency")

args = parser.parse_args()
init_environment(args.seed, args.gpu, cudnn=True, benchmark=True, deterministic=False)

similar_loss = lambda x, y: metric.lncc_loss(x, y, kernel_size=9, eps=1e-9)
smooth_loss = metric.spatial_grad_loss

# --------   Configure        ----------------------------------------------------

voxel_size = [1.0, 1.0, 1.0]
spatial_size = [160, 192, 224]
label_shape = [1, 35, *spatial_size]
save_index_keys = ["similar", "dice", "loss"]

log_path = os.path.join(f"./logs/OASIS-smooth={args.lmd_smooth}-dice={args.lmd_dice:.1f}", time.strftime("%Y-%m%d-%H%M", time.localtime()))
checkpoint_path = f"{log_path}/checkpoint"
tensorboard_path = f"{log_path}/tensorboard"
visualize_path = f"{log_path}/visualize"
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(tensorboard_path, exist_ok=True)
os.makedirs(visualize_path, exist_ok=True)
backup_files([__file__, "datasets.py", "core", "geometry.py", "utils.py", "metric.py"], log_path)

print(f"PID: {os.getpid()}")
print(f"Training on: OASIS")
print(f"Logging path: {log_path}")
print()


print("---------------------------- Parameters -----------------------------------------")
print(format_dict(vars(args), ncols=8))
print("---------------------------------------------------------------------------------")
print()


# --------------------------------------------------------------------------------------

from core.ShiftMorph import ShiftMorph
model = ShiftMorph(3, 16, diff=0, scale=0.3, gamma=0.01, flip=True, share=True)

# --------------------------------------------------------------------------------------


model = model.cuda()
stn = Morpher().cuda()

# optimizer
optimizer = smart_optimizer(model, "AdamW", lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=[
        partial(cosine_annealing_cycle, T0=args.T0, Tmul=args.Tmul, f_min=1e-2),
        partial(cosine_annealing_cycle, T0=args.max_epoch, Tmul=1, f_min=1e-2),
    ],
)
scaler = torch.GradScaler()

# logger
global_step = 0
best_info = {key: float("inf") for key in save_index_keys}
tbwriter = SummaryWriter(tensorboard_path)

train_data = OASISDataset().select("train")
val_data = OASISDataset().select("val")
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)


@torch.autocast(device_type="cuda", enabled=args.amp)
def forward(model, fix_img, mov_img, fix_seg=None, mov_seg=None, exhaust_mode=False):
    loss_info = {}
    loss = 0.0
    flow = model(fix_img, mov_img)

    if args.lmd_sim > 0.0 or exhaust_mode:
        warped_img = stn.warp(mov_img, flow, mode="bilinear")
        similar = similar_loss(fix_img, warped_img)
        loss_info["similar"] = similar.item()
        loss += args.lmd_sim * similar

    if args.lmd_smooth > 0.0 or exhaust_mode:
        smooth = smooth_loss(flow)
        loss_info["smooth"] = smooth.item()
        loss += args.lmd_smooth * smooth

    if args.lmd_dice > 0.0 or exhaust_mode:
        if model.training:
            warped_seg = stn.warp(mov_seg, flow, mode="bilinear")
        else:
            warped_seg = stn.warp(mov_seg, flow, mode="nearest")
        dice = metric.dice_loss(fix_seg, warped_seg)
        loss_info["dice"] = dice.item()
        loss += args.lmd_dice * dice

    loss_info["Jac"] = metric.neg_Jacdet_percent(flow).item()
    loss_info["loss"] = loss.item()
    return flow, warped_img.float(), loss, loss_info


def train_one_epoch(epoch):
    accumulate = Accumulator()
    global global_step
    model.train()
    start_time = time.time()
    for step, batch_data in enumerate(train_loader):
        global_step += 1

        fix_img = batch_data["fixed_image"].cuda().float()
        mov_img = batch_data["moving_image"].cuda().float()
        fix_seg = mov_seg = None

        if args.lmd_dice > 0:
            fix_seg = index2onehot(batch_data["fixed_label"].cuda(), shape=label_shape).float()
            mov_seg = index2onehot(batch_data["moving_label"].cuda(), shape=label_shape).float()

        optimizer.zero_grad()
        flow, warped_img, loss, loss_info = forward(model, fix_img, mov_img, fix_seg, mov_seg, exhaust_mode=False)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accumulate.append(loss_info)

        if step % args.print_freq == 0:
            info = f"--- |  train {epoch:04d}/{step:04d}-> " + format_dict(loss_info)
            print(info)

    scheduler.step()
    loss_mean = accumulate.average()
    loss_mean["lr"] = optimizer.param_groups[0]["lr"]
    tbadd_dict(tbwriter, loss_mean, global_step, group="train")
    loss_mean["time"] = "%.2f min" % ((time.time() - start_time) / 60)
    print(f"*train epoch {epoch:04d} mean-> " + format_dict(loss_mean))


@torch.no_grad()
def evaluate(epoch):
    model.eval()
    accumulate = Accumulator()
    start_time = time.time()

    for step, batch_data in enumerate(val_loader):

        fix_img = batch_data["fixed_image"].cuda().float()
        fix_seg = index2onehot(batch_data["fixed_label"].cuda(), shape=label_shape).float()

        mov_img = batch_data["moving_image"].cuda().float()
        mov_seg = index2onehot(batch_data["moving_label"].cuda(), shape=label_shape).float()
        flow, warped_img, loss, loss_info = forward(model, fix_img, mov_img, fix_seg, mov_seg, exhaust_mode=True)

        fig, axes = plt.subplots(2, 3, figsize=(2, 1))
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.01, hspace=0.01)
        three_view(fix_img.cpu(), ax=axes[0], cmap="Blues", enhance=False, alpha=0.8)
        three_view(mov_img.cpu(), ax=axes[0], cmap="Oranges", enhance=False, alpha=0.5)
        three_view(fix_img.cpu(), ax=axes[1], cmap="Blues", enhance=False, alpha=0.8)
        three_view(warped_img.cpu(), ax=axes[1], cmap="Oranges", enhance=False, alpha=0.5)
        fig.savefig(f"{visualize_path}/{step:02d}.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
        accumulate.append(loss_info)
        plt.close(fig)

    loss_mean = accumulate.average()
    tbadd_dict(tbwriter, loss_mean, epoch, group="val")
    loss_mean["time"] = "%.2f min" % ((time.time() - start_time) / 60)
    print(f"#Test  epoch {epoch:04d} mean-> " + format_dict(loss_mean))
    return loss_mean


for epoch in range(args.max_epoch):
    train_one_epoch(epoch)
    loss_info = evaluate(epoch)
    checkpoints = dict(
        epoch=epoch,
        net=model.state_dict(),
    )

    torch.save(checkpoints, f"{checkpoint_path}/recent.ckpt")
    for key in save_index_keys:
        cur = loss_info[key]
        prev = best_info[key]
        if cur < prev:
            best_info[key] = cur
            torch.save(checkpoints, f"{checkpoint_path}/best_{key}.ckpt")
            print(f"Best {key}! epoch {epoch:04d}: previous = {prev:.6f} current = {cur:.6f}")
    print()

tbwriter.close()
