# _*_ coding: utf-8 _*_
"""
Time:     2022-05-05 13:40
Author:   Haolin Yan(XiDian University)
File:     check.py.py
"""
from mmt.converter import validation
from mmt.predictor import latency_predictor
validation("mbv3_train", "mbv3_ops/meta_latency.pkl", save_path="train_error.csv")
lp = latency_predictor("mbv3_ops", "train_error.csv")
validation("mbv3", "mbv3_ops/meta_latency.pkl", save_path="gp_error.csv", lp=lp)
validation("mbv3", "mbv3_ops/meta_latency.pkl", save_path="error.csv")
