#!/usr/bin/env bash
cd /nfs/project/libo_i/mmdetection
python3 tools/test.py configs/coco/mask_rcnn_r50_fpn_gn_2x.py work_dirs/mask_rcnn_r50_fpn_gn_2x/epoch_18.pth --out val.pkl --gpus 4 --eval bbox segm


