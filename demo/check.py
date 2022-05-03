# _*_ coding: utf-8 _*_
"""
Time:     2022-05-03 17:20
Author:   Haolin Yan(XiDian University)
File:     check.py.py
"""
import sys
from resnet18 import ResNet18
import logging
import copy
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.append("/tmp/pycharm_project_937")
from core.parser import predict_latency
from core.meter import get_model_latency


def generate_different_cfg():
    configs = []
    kernel = [3, 5]
    cfg = [0] * 8

    def fill_(cfg, index):
        if index == 8:
            configs.append(cfg)
            return
        for k in kernel:
            cfg_ = copy.deepcopy(cfg)
            cfg_[index] = k
            fill_(cfg_, index + 1)

    fill_(cfg, 0)
    return configs


configs = generate_different_cfg()
path = "ops_resnet18/meta_latency.pkl"
pred = []
true = []

for i, c in enumerate(configs):
    model = ResNet18(c)
    pred.append(predict_latency(model, path, [1, 3, 224, 224], verbose=False))
    true.append(get_model_latency(model, [1, 3, 224, 224])["Avg"])
    print("(%d/%d) pred: %f true: %f" % (i, len(configs), pred[i], true[i]))

pd.DataFrame({"pred": pred, "true": true}).to_csv("check_result.csv", index=False)



