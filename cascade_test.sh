#!/usr/bin/env bash
python3 tools/test.py configs/cascade_mask_rcnn_x101_64x4d_fpn_1x.py models/cascade_mask_rcnn_x101_64x4d_fpn_20e_20181218-630773a7.pth --gpus 1 --out cascade_X101.pkl --eval bbox segm --proc_per_gpu 16

