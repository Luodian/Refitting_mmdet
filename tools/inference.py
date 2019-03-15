import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result
from mmdet.models import build_detector

# import mmdet
# print(mmdet.__file__)

cfg = mmcv.Config.fromfile('/nfs/project/libo_i/mmdetection/configs/mask_rcnn_r50_fpn_gn_2x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, '/nfs/project/libo_i/mmdetection/models/mask_rcnn_r50_fpn_gn_2x_20180113-86832cf2.pth')

# test a list of images
imgs = ['imgs/test5.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    print(i, imgs[i])
    show_result(imgs[i], result, outfile="infered_{}".format(imgs[i]))
