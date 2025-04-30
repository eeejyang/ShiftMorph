# Medical Image Registration with ShiftMorph

This repository contains the implementation of [ShiftMorph, a fast and robust convolutional neural network for 3D deformable medical image registration](https://dl.acm.org/doi/abs/10.1145/3664647.3680828), as presented in our ACM MM 2024 paper.


## Train [OASIS](https://learn2reg.grand-challenge.org/Datasets/) Brain MRI Patient-to-Patient Registration
``` bash
python -u 1_train_oasis.py \
    --amp --gpu=0 \
    --lmd_sim=1 --lmd_smooth=1 --lmd_dice=0 \
    --max_epoch=300 --T0=300 --Tmul=1 \
    --print_freq=200
```

## Train [IXI](https://brain-development.org/ixi-dataset/) Brain MRI Atlas-to-Patient Registration
``` bash
python -u 1_train_ixi.py \
    --amp --gpu=0 \
    --lmd_sim=1 --lmd_smooth=4 --lmd_dice=0 \
    --max_epoch=300 --T0=300 --Tmul=1 \
    --print_freq=200
```

## Citation
If you find this code useful for your research, please cite our paper:

``` tex
@inproceedings{shiftMorph2024yang,
  title = {ShiftMorph: A Fast and Robust Convolutional Neural Network for 3D Deformable Medical Image Registration},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  author = {Lijian, Yang and Weisheng, Li and Yucheng, Shu and Jianxun, Mi and Yuping, Huang and Bin, Xiao},
  year = {2024},
  pages = {2814--2823},
  doi = {10.1145/3664647.3680828},
  location = {Melbourne VIC, Australia},
}
```
