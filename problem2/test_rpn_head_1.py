from mmengine.runner import Runner
from mmdet.apis import inference_detector, init_detector
import mmcv
from mmdet.structures import DetDataSample
from mmengine.config import Config
from mmengine.dataset import Compose
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mmcv.visualization import imshow_det_bboxes

# 加载 config
config_file = './voc_detection_mask_rcnn.py'
checkpoint_file = '../work_dir/mask_rcnn/epoch_5.pth'
# img_file = '/work/jpma/chenxr/homework/mmdetection/data/VOCdevkit/coco/test2017/2007_001423.jpg'
img_file = '/work/jpma/chenxr/homework/mmdetection/A_homework/test_set/person.jpg'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

cfg = Config.fromfile(config_file)
test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline)

data = dict(img_path=img_file, img_id=0)
data = test_pipeline(data)
data['inputs'] = data['inputs'].unsqueeze(0)
data['data_samples'] = [data['data_samples']]
data_processed = model.data_preprocessor(data, False)

with torch.no_grad():
    features = model.extract_feat(data_processed['inputs'])
    proposals = model.rpn_head.predict(features, data_processed['data_samples'], rescale=False)
    scores = proposals[0].scores.cpu().numpy()
    bboxes = proposals[0].bboxes.cpu().numpy()
    labels = proposals[0].labels.cpu().numpy()

img_model = data['inputs'].squeeze(0).permute(1,2,0).cpu().numpy()

imshow_det_bboxes(
    img_model,
    bboxes[:10],
    scores[:10],
    show=False,
    out_file='rpn_boxes.jpg'
)
mmcv.imwrite(img_model, 'processed_img.jpg')