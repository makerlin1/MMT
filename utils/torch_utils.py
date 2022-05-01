# _*_ coding: utf-8 _*_
"""
Time:     2022-05-01 17:23
Author:   Haolin Yan(XiDian University)
File:     torch_utils.py
"""
import importlib
import yaml
import copy
import logging
import torch
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def import_module(pkg):
    return importlib.import_module(pkg)


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


def parse_yaml(path):
    """
    Parse the YAML file and return all operators
    e.g.
        parse_yaml("docs/demo.yaml")
    """
    ops_register = []
    with open(path, encoding='utf-8') as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    for k, v in data.items():
        if v == "None":
            continue
        # import package
        pkg = import_module(k)
        for ops, param in v.items():
            if param == "None":
                ops_register.append((getattr(pkg, ops)(), [1, 1]))
                continue
            cfg_list = generate_param_list(param)
            for cfg in cfg_list:
                input_shape = cfg["input_shape"]
                del cfg["input_shape"]
                ops_register.append((getattr(pkg, ops)(**cfg), input_shape))
    return ops_register


def parser_model(module, ops_list, verbose=False):
    """
    Parse the operators in the model and return the corresponding dictionary
    e.g.
        ops_list = [MLP, nn.Linear, nn.ReLU]  # MLP is designed module by yourself
        arch2ops = parser_model(model, ops_list, verbose=True)
    """
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


def convert2mnn(ops, path, verbose=False):
    model, input_shape = ops
    model.eval().cuda()
    dummy_input = torch.randn(input_shape, device='cuda')
    input_names = ["input"]
    output_names = ["output"]
    name = os.path.join(path, replace(model.__repr__()))
    onnx_name = name + ".onnx"
    mnn_name = name + ".mnn"
    torch.onnx.export(model, dummy_input, onnx_name, verbose=verbose, input_names=input_names,
                      output_names=output_names)
    logger.info("Export %s to %s" % (name, onnx_name))
    cmd = "mnnconvert -f ONNX --modelFile {} --MNNModel {} --bizCode biz".format(onnx_name, mnn_name)
    os.system(cmd)
    if not os.path.isfile(mnn_name):
        raise ValueError("Fail to convert %s to %s" % (onnx_name, mnn_name))
    logger.info("Convert %s to %s" % (onnx_name, mnn_name))

