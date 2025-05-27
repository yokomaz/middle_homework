from mmdet.apis import init_detector, inference_detector, DetInferencer
from mmdet.apis import init_detector, inference_detector
import torch
import matplotlib.pyplot as plt
import cv2
import mmcv
import matplotlib.patches as patches

# 指定模型的配置文件和 checkpoint 文件路径
config_file = '../configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '../work_dirs/epoch_12.pth'
#img = '/work/jpma/chenxr/homework/mmdetection/data/VOCdevkit/coco/test2017/2007_001698.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
img = '/work/jpma/chenxr/homework/mmdetection/A_homework/test_set/train.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device='cpu')
result = inferencer(img, out_dir='./results_sparse', pred_score_thr=0.3, )
