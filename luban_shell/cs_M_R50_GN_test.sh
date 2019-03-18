#!/usr/bin/env bash
cd /nfs/project/libo_i/mmdetection
python3 tools/test.py configs/cs_mask_rcnn_r50_fpn_gn_2x.py work_dirs/cityscapes_mask_rcnn_r50_fpn_gn_2x/epoch_24.pth --gpus 1 --proc_per_gpu 8 --out CS_mask_rcnn_R50_gn.pkl --eval bbox segm