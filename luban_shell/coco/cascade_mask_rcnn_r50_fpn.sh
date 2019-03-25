#!/usr/bin/env bash
cd /nfs/project/libo_i/mmdetection
pkill python3
pkill python
python3 tools/train.py configs/coco/cascade_mask_rcnn_r50_fpn_1x.py 4 --validate