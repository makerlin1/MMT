# _*_ coding: utf-8 _*_
"""
Time:     2022-05-04 22:23
Author:   Haolin Yan(XiDian University)
File:     convert.py
"""
from mmt import register
import torch.nn as nn
from mobilenetv3 import (conv_3x3_bn,
                         InvertedResidual,
                         conv_1x1_bn,
                         h_swish)
fp = "./mbv3_ops"
reg = lambda ops, **kwargs: register(ops, fp, **kwargs)

reg(conv_3x3_bn,
    inp=[3],
    oup=[16],
    stride=[2],
    input_shape=[[1, 3, 224, 224]])

reg(conv_1x1_bn,
    inp=[96],
    oup=[576],
    input_shape=[[1, 96, 7, 7]])

reg(InvertedResidual,
    inp=[16, 16],
    hidden_dim=[16, 72],
    oup=[16, 24],
    kernel_size=[3, 5, 7],
    stride=[1, 2],
    use_se=[1, 0],
    use_hs=[1, 0],
    input_shape=[[1, 16, 112, 112], [1, 16, 56, 56]]
    )

reg(InvertedResidual,
    inp=[24],
    hidden_dim=[88, 96],
    oup=[24, 40],
    kernel_size=[3, 5, 7],
    stride=[1, 2],
    use_se=[1, 0],
    use_hs=[1, 0],
    input_shape=[[1, 24, 28, 28]]
    )

reg(h_swish,
    no_params=True,
    input_shape=[[1, 1024]]
)

reg(InvertedResidual,
    inp=[40],
    hidden_dim=[240, 120],
    oup=[40, 48],
    kernel_size=[3, 5, 7],
    stride=[1],
    use_se=[1, 0],
    use_hs=[1, 0],
    input_shape=[[1, 40, 14, 14]]
    )

reg(InvertedResidual,
    inp=[48],
    hidden_dim=[144, 288],
    oup=[48, 96],
    kernel_size=[3, 5, 7],
    stride=[1, 2],
    use_se=[1, 0],
    use_hs=[1, 0],
    input_shape=[[1, 48, 14, 14]]
    )

reg(InvertedResidual,
    inp=[96],
    hidden_dim=[576],
    oup=[96, 96],
    kernel_size=[3, 5, 7],
    stride=[1, 2],
    use_se=[1, 0],
    use_hs=[1, 0],
    input_shape=[[1, 96, 7, 7]]
    )

reg(nn.Linear,
    in_features=[576, 1024],
    out_features=[1024, 1000],
    bias=[True],
    input_shape=[[1, 576], [1, 1024]],
    )











