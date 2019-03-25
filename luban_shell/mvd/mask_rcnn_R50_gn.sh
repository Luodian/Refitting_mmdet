#!/usr/bin/env bash
cd /nfs/project/libo_i/mmdetection
./tools/dist_train.sh configs/mapillary/mask_rcnn_r50_fpn_1x.py 4 --validate