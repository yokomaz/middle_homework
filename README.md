# 这是复旦大学大数据学院张力老师的研究生课程：神经网络与深度学习 2025年度上半学期期中作业

problem1为任务1：在ImageNet上预训练的卷积神经网络实现CalTech101分类。

problem2为任务2：使用mmdetection目标检测框架，在VOC数据集上训练并测试模型Mask R-CNN与Sparse R-CNN

训练好的网络权重在https://drive.google.com/drive/folders/1n7jWwbzueo9ESeArs1OcIla9X_VbNh7v?usp=drive_link

# 任务1使用方法：

主要包含main.py与test.py

## 训练方法

1. 修改main.py中的学习率列表，批次大小列表内参数或保持原样以执行网格搜索。2. 修改数据集路径以适配对应数据集。随后执行python main.py以进行训练

## 测试方法

1. 修改test.py中的数据集路径，随后执行test.py进行测试，可测试的模型有：best_model_unpretrained.pth与best_model_pretrained.pth

# 任务2使用方法：

主要包含1. mmdetection框架下有关Mask R-CNN与Sparse R-CNN的训练配置文件（voc_detection_mask_rcnn.py & sparse-rcnn_r50_fpn_1x_coco.py） 2. 用于将VOC数据转化为COCO数据集格式的脚本（voc2coco.py） 3. 在转化后VOC数据集中图片添加mask标注的脚本（get_mask.py） 4. 用于训练的配置文件
5. 用于可视化结果的脚本（vis_result_mask_rcnn.py & voc_detection_mask_rcnn.py & vis_result_sparse_rcnn.py & test_rpn_head.py）

## 推理inference方法

使用vis_result_mask_rcnn.py或vis_result_sparse_rcnn.py，修改其中权重参数，配置文件以及需要推理的图片路径，执行python xxx.py进行推理

## 可视化Mask R-CNN proposal box方法

使用test_rpn_head_1.py，修改权重参数文件，配置文件路径以及需要推理的图片路径，执行python test_rpn_head_1.py查看Mask R-CNN第一阶段proposal boxes
