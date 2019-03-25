#!/usr/bin/env bash
cd /nfs/project/libo_i/mmdetection
python3 selfmade/batch_test_all_epoch.py cascade_mask_rcnn_x101_64x4d_fpn_1x cityscapes > luban_shell/cityscapes/cs_cascade_X101_seq_test.txt
