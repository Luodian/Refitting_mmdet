#!/usr/bin/env bash
python3 tools/test.py configs/cascade_mask_rcnn_x101_64x4d_fpn_1x.py work_dirs/cascade_mask_rcnn_x101_64x4d_fpn_1x/epoch_5.pth --gpus 4 --out cascade_X101.pkl --eval bbox segm --proc_per_gpu 16
