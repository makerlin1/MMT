# _*_ coding: utf-8 _*_
"""
Time:     2022-05-03 0:28
Author:   Haolin Yan(XiDian University)
File:     main.py.py
"""
import sys
sys.path.append("/tmp/pycharm_project_937")
import mmt.parser as parser
import torch.nn as nn
from mmt.parser import summary_model
from resnet18 import ResNet18, ResNetBasicBlock, ResNetDownBlock
model = ResNet18([3,3,3,3,3,3,3,3])
ops_list = [ResNetBasicBlock, ResNetDownBlock, nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear]
summary_model(model, [1, 3, 224, 224], ops_list)
