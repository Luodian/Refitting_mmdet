#!/usr/bin/env bash
cd /nfs/project/libo_i/mmdetection
python3 tools/test.py configs/KITTI/cascade_mask_rcnn_x101_64x4d_fpn_1x.py work_dirs/kitti_cascade_mask_rcnn_x101_64x4d_fpn_1x/epoch_8.pth --out kt_val.pkl --gpus 4 --eval bbox segm


