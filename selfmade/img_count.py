import json
import os

tasks = ['train', 'test', 'val']


def validate_img():
	data_root = "/nfs/project/data/cityscapes/leftImg8bit/"
	for item in tasks:
		extended_data_path = os.path.join(data_root, item)
		print(extended_data_path)
		fs = os.listdir(extended_data_path)
		cnt = 0
		for sub_item in fs:
			sub_item_path = os.path.join(extended_data_path, sub_item)
			if not os.path.isdir(sub_item_path):
				# print(sub_item_path)
				cnt += 1
		print("{} images in {}".format(cnt, item))


coco = "/nfs/project/libo_i/mmdetection/data/coco/annotations/instances_val2017.json"
cityscapes = "/nfs/project/libo_i/mmdetection/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
kitti = "/nfs/project/libo_i/mmdetection/data/kitti/annotations/training.json"


def validate_ann():
	base_path = kitti
	with open(base_path, 'r') as fp:
		ann = json.load(fp)
	print(len(ann['annotations']))
	print(len(ann['images']))


validate_ann()
