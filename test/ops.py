# _*_ coding: utf-8 _*_
"""
Time:     2022-05-01 17:06
Author:   Haolin Yan(XiDian University)
File:     ops.py.py
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_c, hid_c),
                                nn.ReLU(),
                                nn.Linear(hid_c, out_c))

    def forward(self, x):
        return self.fc(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mlp = MLP(12, 23, 12)
        self.mlp2 = MLP(12, 23, 12)

    def forward(self, x):
        return self.mlp2(self.mlp(x))


def parser_model(module, ops_list, verbose=False):
    arch2ops = {}

    def search_ops(module, ops_list):
        for name, m in module.named_children():
            found = False
            for ops in ops_list:
                if isinstance(m, ops):
                    if verbose:
                        print("{} belongs to {}".format(name, ops.__name__))
                    found = True
                    arch2ops[name] = ops.__name__
                    break
            if not found:
                search_ops(m, ops_list)
    search_ops(module, ops_list)
    return arch2ops


if __name__ == "__main__":
    model = Model()
    ops_list = [MLP, nn.Linear, nn.ReLU]
    arch2ops = parser_model(model, ops_list, verbose=True)
    print(arch2ops)
