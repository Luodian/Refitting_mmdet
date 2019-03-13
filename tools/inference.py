import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result
from mmdet.models import build_detector

cfg = mmcv.Config.fromfile('/nfs/project/libo_i/mmdetection/configs/cascade_mask_rcnn_x101_64x4d_fpn_1x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg = cfg.test_cfg)
_ = load_checkpoint(model, '/nfs/project/libo_i/mmdetection/models/cascade_mask_rcnn_x101_64x4d_fpn_20e_20181218-630773a7.pth')

# test a list of images
imgs = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device = 'cuda:0')):
	print(i, imgs[i])
	show_result(imgs[i], result)
