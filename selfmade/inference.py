import os

import mmcv
import numpy as np
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result
from mmdet.models import build_detector


def kitti():
	cfg = mmcv.Config.fromfile('/nfs/project/libo_i/mmdetection/configs/KITTI/cascade_mask_rcnn_x101_64x4d_fpn_1x.py')
	cfg.model.pretrained = None
	
	# construct the model and load checkpoint
	model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
	_ = load_checkpoint(model, '/nfs/project/libo_i/mmdetection/work_dirs/kitti_cascade_mask_rcnn_x101_64x4d_fpn_1x/epoch_24.pth')
	
	# test a list of images
	img_path = '/nfs/project/libo_i/mmdetection/data/kitti/testing/kitti_demo_image'
	fs = os.listdir(img_path)
	imgs = []
	for item in fs:
		item_path = os.path.join(img_path, item)
		if not os.path.isdir(item_path):
			imgs.append(item_path)
	
	for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
		print(i, imgs[i])
		save_path = "/nfs/project/libo_i/mmdetection/selfmade/kitti_demo_image/24_kitti_inferred/{}".format(os.path.basename(imgs[i]))
		print(save_path)
		show_result(imgs[i], result, dataset='kitti', outfile=save_path)

kitti()
