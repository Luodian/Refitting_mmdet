#!/usr/bin/env bash
cd /nfs/project/libo_i/mmdetection
python3 tools/train.py ./configs/cityscapes/cascade_mask_rcnn_x101_64x4d_fpn_1x.py --gpus 4