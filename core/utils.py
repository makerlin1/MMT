# _*_ coding: utf-8 _*_
"""
Time:     2022-05-01 17:23
Author:   Haolin Yan(XiDian University)
File:     utils.py
"""
import importlib
import copy


class ops_meta:
    def __init__(self, classesname, input_shape, init_param=None):
        self.classname = classesname
        self.input_shape = input_shape
        self.init_param = init_param
        self.mnn_fname = None
        self.avg = 0
        self.max = 0
        self.min = 0

    def __repr__(self):
        return "%s-[%s]-%s" % (self.classname, shape2str(self.input_shape), self.mnn_fname)

    def return_instance(self):
        if self.init_param:
            return self.classname(**self.init_param)
        else:
            return self.classname()

    def record_mnn_fname(self, fname):
        self.mnn_fname = fname

    def record_mnn_performance(self, result):
        self.avg = result["Avg"]
        self.min = result["Min"]
        self.max = result["Max"]


def import_module(pkg):
    return importlib.import_module(pkg)


def shape2str(shape):
    s = ""
    for i in shape:
        s += str(i) + '-'
    return s[:-1]


def generate_param_list(kwargs):
    cfg_list = []

    def generate_param(param, **kwargs):
        for i, (k, v) in enumerate(param.items()):
            if v or k == "input_shape":
                continue
            table = kwargs[k]
            for j, ele in enumerate(table):
                param[k] = ele
                param_ = copy.deepcopy(param)
                if i == 0:
                    param_["input_shape"] = kwargs["input_shape"][j]
                generate_param(param_, **kwargs)
            return 0
        cfg_list.append(param)

    param = dict((n, None) for n in kwargs.keys())
    generate_param(param, **kwargs)
    return cfg_list


def remove_(str):
    last = ""
    new_str = ""
    for s in str:
        if s == last:
            continue
        new_str += s
        last = s
    return new_str


def replace(str):
    str = str.replace("(", "-").replace(")", "-").replace(",", "-").replace(" ", "")
    return remove_(str)

