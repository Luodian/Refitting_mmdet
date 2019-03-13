import cv2
import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector
from mmdet.models import build_detector
from mmdet.apis import show_result
from vis_utils import vis

# import mmdet
# print(mmdet.__file__)

cfg = mmcv.Config.fromfile('/nfs/project/libo_i/mmdetection/configs/cascade_mask_rcnn_x101_64x4d_fpn_1x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg = cfg.test_cfg)
_ = load_checkpoint(model, '/nfs/project/libo_i/mmdetection/models/cascade_mask_rcnn_x101_64x4d_fpn_20e_20181218-630773a7.pth')

# test a list of images
imgs = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device = 'cuda:0')):
	print(i, imgs[i])
	im_name = imgs[i]
	im = cv2.imread(im_name)
	cls_boxes = result[0]
	cls_segms = result[1]
	vis.vis_one_image(
		im[:, :, ::-1],  # BGR -> RGB for visualization
		im_name,
		"/nfs/project/libo_i/mmdetection/infer_img",
		cls_boxes,
		cls_segms,
		box_alpha = 0.3,
		show_class = False,
		thresh = 0.7,
		kp_thresh = 2
	)
	# show_result(imgs[i], result, outfile = "infered_{}".format(imgs[i]))
