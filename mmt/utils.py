# _*_ coding: utf-8 _*_
"""
Time:     2022-05-01 17:23
Author:   Haolin Yan(XiDian University)
File:     utils.py
"""
import importlib
import copy
import pickle
import os
import json
from .meter import get_latency
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ops_meta:
    def __init__(self, classesname, input_shape, init_param=None):
        self.classname = classesname
        self.input_shape = input_shape
        self.init_param = init_param
        self.mnn_fname = None
        self.avg = 0
        self.max = 0
        self.min = 0
        self.repr = ""
        self.set_path = ""

    def __repr__(self):
        return self.repr

    def return_instance(self):
        if self.init_param:
            model = self.classname(**self.init_param)
        else:
            model = self.classname()
        self.repr = model.__repr__()
        return model

    def record_mnn_fname(self, fname):
        mnn2cls = fname
        with open(os.path.join(self.set_path, mnn2cls+".pj"), 'wb') as f:
            pickle.dump(self.classname, f)
            logger.info("write %s" % mnn2cls+".pj")

        delattr(self, 'classname')
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
    index = 0
    key_list = list(kwargs.keys())

    def generate_param(param, index, **kwargs):
        # print(param, index)
        key = key_list[index]
        if key == "input_shape":
            cfg_list.append(param)
            return 0

        table = kwargs[key]
        for j, ele in enumerate(table):
            param[key] = ele
            param_ = copy.deepcopy(param)
            if index == 0:
                param_["input_shape"] = kwargs["input_shape"][j]
            generate_param(param_, index + 1, **kwargs)

    param = dict((n, None) for n in kwargs.keys())
    generate_param(param, index, **kwargs)
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


def parse_ops_info(path):
    ops_info_list = []
    fp = os.path.dirname(path)
    with open(path, "rb") as f:
        ops_list = pickle.load(f)
        for ops in ops_list:
            with open(os.path.join(fp, ops.mnn_fname + '.pj'), "rb") as f:
                class_name = pickle.load(f)
            ops_info_list.append((class_name, ops.input_shape, ops.avg, ops.min, ops.max, ops.repr))
    return ops_info_list


def get_net_device(model):
    return next(model.parameters()).device


def export_models(model, input_shape, path, **kwargs):
    from .converter import convert2mnn
    net = model(**kwargs)
    convert2mnn(net, input_shape, path)
    kwargs["fname"] = path
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(kwargs, f)










