import torch
import numpy as np
import os
import json
import pandas as pd
import glob
from utils import *


class IXIDataset:

    def __init__(self):
        root_dir = "/home/yanglj/Desktop/datasets/IXI/"
        train_list = sorted(glob.glob(f"{root_dir}/train/*.pkl"))
        val_list = sorted(glob.glob(f"{root_dir}/val/*.pkl"))
        test_list = sorted(glob.glob(f"{root_dir}/test/*.pkl"))

        self.file_dict = dict(train=train_list, test=test_list, val=val_list)
        self.num_class = 30
        self.state = "train"
        self.selected_files = self.file_dict[self.state]
        self.atlas, self.atlas_seg = pkload(f"{root_dir}/atlas_onehot.pkl")

    def select(self, state="train"):
        self.selected_files = self.file_dict[state]
        self.state = state
        return self

    def __len__(self):
        return len(self.selected_files)

    def __getitem__(self, idx):
        path = self.selected_files[idx]
        x, x_seg = pkload(path)
        dict_item = {
            "fixed_name": os.path.basename(path),
            "moving_name": "atlas.pkl",
            "fixed_image": x[None, ...],
            "fixed_label": x_seg,
            "moving_image": self.atlas[None, ...],
            "moving_label": self.atlas_seg,
        }
        return dict_item


class OASISDataset:

    def __init__(self):
        root_dir = "/home/yanglj/Desktop/datasets/OASIS/imgs/"

        train_list = sorted(glob.glob(f"{root_dir}/train/*.pkl"))
        val_list = sorted(glob.glob(f"{root_dir}/val/*.pkl"))
        test_list = sorted(glob.glob(f"{root_dir}/test/*.pkl"))
        self.file_dict = dict(train=train_list, test=test_list, val=val_list)
        self.state = "train"
        self.num_class = 35
        self.selected_files = self.file_dict[self.state]

    def select(self, state="train"):
        self.selected_files = self.file_dict[state]
        self.state = state
        return self

    def __len__(self):
        return len(self.selected_files)

    def __getitem__(self, idx):
        length = len(self)
        shift = np.random.randint(1, length) if self.state == "train" else 1
        path1 = self.selected_files[idx]
        path2 = self.selected_files[(idx + shift) % length]
        x, x_seg = pkload(path1)
        y, y_seg = pkload(path2)
        dict_item = {
            "fixed_name": os.path.basename(path1),
            "moving_name": os.path.basename(path2),
            "fixed_image": x[None, ...],
            "fixed_label": x_seg,
            "moving_image": y[None, ...],
            "moving_label": y_seg,
        }
        return dict_item
