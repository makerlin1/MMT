# _*_ coding: utf-8 _*_
"""
Time:     2022-05-05 13:40
Author:   Haolin Yan(XiDian University)
File:     check.py.py
"""
from mmt.converter import validation
validation("mbv3", "mbv3_ops/meta_latency.pkl", save_path="error.txt")