#!/usr/bin/env bash
cd /nfs/project/libo_i/mmdetection
python3 tools/test.py configs/cityscapes/mask_rcnn_r50_fpn_1x.py /nfs/project/libo_i/mmdetection/work_dirs/cityscapes_mask_rcnn_r50_fpn_1x/epoch_12.pth --gpus 1 --proc_per_gpu 8 --out CS_mask_rcnn_R50.pkl --eval bbox segm