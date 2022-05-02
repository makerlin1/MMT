# _*_ coding: utf-8 _*_
"""
Time:     2022-05-03 0:28
Author:   Haolin Yan(XiDian University)
File:     main.py.py
"""
import sys
sys.path.append("/tmp/pycharm_project_937")
# Convert
from core.converter import generate_ops_list
generate_ops_list("resnet18.yaml", "ops_resnet18")
# Measure
from core.meter import meter_ops
meter_ops("ops_resnet18", times=100)
