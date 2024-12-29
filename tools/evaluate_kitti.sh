#!/bin/bash


KITTI_PATH=datasets/kitti/dataset/sequences/$seq

evalset=(
    00
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH/$seq --weights=droid.pth --disable_vis $@
done

