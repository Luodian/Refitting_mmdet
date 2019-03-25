import os

import cv2
import mmcv
import numpy as np
import pycocotools.mask as mask_util
import scipy.misc as msc
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector
from mmdet.models import build_detector

num_images = 200
kitti_cls_num = 10
pred_list_name = 'pred_list'
pred_img_name = 'pred_img'

test_image_path = "/nfs/project/libo_i/mmdetection/data/kitti/testing/image_2"

test_image_list = os.listdir(test_image_path)

output_dir = "/nfs/project/libo_i/mmdetection/selfmade/kt_submission"
pred_list_path = os.path.join(output_dir, pred_list_name)
pred_img_path = os.path.join(output_dir, pred_img_name)

# Ensure path exists.
if not os.path.exists(pred_list_path):
	os.makedirs(pred_list_path)

if not os.path.exists(pred_img_path):
	os.makedirs(pred_img_path)

cfg = mmcv.Config.fromfile('/nfs/project/libo_i/mmdetection/configs/KITTI/cascade_mask_rcnn_x101_64x4d_fpn_1x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, '/nfs/project/libo_i/mmdetection/work_dirs/kitti_cascade_mask_rcnn_x101_64x4d_fpn_1x/epoch_24.pth')

for ind, result in enumerate(
	inference_detector(model, list(map(lambda item: os.path.join(test_image_path, item), test_image_list)), cfg, device='cuda:0')):
	text_save_name = "{}.txt".format(test_image_list[ind][:-4])
	text_save_path = os.path.join(pred_list_path, text_save_name)
	file = open(text_save_path, "w")
	bbox = result[0]
	segm = result[1]
	ist_cnt = 0
	for cat_id in range(kitti_cls_num):
		for sub_bbox, sub_segm in zip(bbox[cat_id], segm[cat_id]):
			score = sub_bbox[4]
			if score < 0.3:
				continue
			mask = np.array(mask_util.decode(sub_segm), dtype=np.float32)
			im = cv2.imread(os.path.join(test_image_path, test_image_list[ind]))
			
			instances_graph = np.zeros((im.shape[0], im.shape[1]))
			instances_graph[mask == 1] = 255
			instance_save_name = "{}_{:0>3d}.png".format(test_image_list[ind][:-4], ist_cnt)
			print(instance_save_name)
			
			instance_save_path = os.path.join(pred_img_path, instance_save_name)
			msc.imsave(instance_save_path, instances_graph)
			
			instance_info_To_Text = "../pred_img/{} {:0>3d} {}\n".format(instance_save_name, cat_id + 24, score)
			file.writelines(instance_info_To_Text)
			ist_cnt += 1
	file.close()

import zipfile

zipped_file_news = output_dir + '.zip'
z = zipfile.ZipFile(zipped_file_news, 'w', zipfile.ZIP_DEFLATED)
for dirpath, dirnames, filenames in os.walk(output_dir):
	fpath = dirpath.replace(output_dir, '')
	fpath = fpath and fpath + os.sep or ''
	for filename in filenames:
		z.write(os.path.join(dirpath, filename), fpath + filename)

print('Done')
z.close()
