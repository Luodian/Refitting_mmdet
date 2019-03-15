# created to split imamges in different cities in cityscapes

import os
import shutil

data_root = "/nfs/project/data/cityscapes/leftImg8bit/"

fs = os.listdir(data_root)

for item in fs:
	item_path = os.path.join(data_root, item)
	if not os.path.isdir(item_path):
		print("Deleting {}".format(item_path))
		os.remove(item_path)

tasks = ['train', 'test', 'val']

for item in tasks:
	extend_data_root = data_root + item
	fs = os.listdir(extend_data_root)
	for subitem in fs:
		subitem_path = os.path.join(extend_data_root, subitem)
		# subitem is each city
		if os.path.isdir(subitem_path):
			imgs = os.listdir(subitem_path)
			for img in imgs:
				img_path = os.path.join(subitem_path, img)
				print("Copying {} to {}".format(img_path, extend_data_root))
				shutil.copy(img_path, extend_data_root)
