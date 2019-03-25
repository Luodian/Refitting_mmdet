#!/usr/bin/env bash
cd /nfs/project/libo_i/mmdetection
python3 selfmade/batch_test_all_epoch.py mask_rcnn_r50_fpn_1x cityscapes > luban_shell/cityscapes/mask_rcnn_r50_fpn_1x.txt
