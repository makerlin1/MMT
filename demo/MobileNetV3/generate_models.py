from mobilenetv3 import MobileNetV3
import random
from mmt.converter import export_models


cfgs = [
    # k, t, c, SE, HS, s
    [3, 1, 16, 1, 0, 2],
    [3, 4.5, 24, 0, 0, 2],
    [3, 3.67, 24, 0, 0, 1],
    [5, 4, 40, 1, 1, 2],
    [5, 6, 40, 1, 1, 1],
    [5, 6, 40, 1, 1, 1],
    [5, 3, 48, 1, 1, 1],
    [5, 3, 48, 1, 1, 1],
    [5, 6, 96, 1, 1, 2],
    [5, 6, 96, 1, 1, 1],
    [5, 6, 96, 1, 1, 1],
]


def generate_cfg(cfgs):
    for c in range(len(cfgs)):
        cfgs[c][0] = random.choice([3, 5, 7])
        cfgs[c][-2] = random.choice([0, 1])
        cfgs[c][-3] = random.choice([0, 1])
    return cfgs


for i in range(200):
    cfg_ = generate_cfg(cfgs)
    net = MobileNetV3(cfg_, id=i, mode="small")
    export_models(net, [1, 3, 224, 224], "mbv3")
