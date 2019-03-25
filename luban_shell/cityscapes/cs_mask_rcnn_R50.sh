#!/usr/bin/env bash
cd /nfs/project/libo_i/mmdetection
python3 tools/train.py ./configs/cityscapes/mask_rcnn_r50_fpn_1x.py --gpus 4 --validate