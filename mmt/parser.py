# _*_ coding: utf-8 _*_
"""
Time:     2022-05-02 13:20
Author:   Haolin Yan(XiDian University)
File:     parser.py
"""
from .utils import (import_module,
                    generate_param_list,
                    ops_meta,
                    parse_ops_info,
                    get_net_device)
import yaml
from tabulate import tabulate
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_latency(module, ops_path, input_shape, verbose=False):
    """Predict the model's latency using the latency table"""
    headers = ["ops", "input_shape", "output_shape", "Avg latency(ms)"]
    ops_list = parse_ops_info(ops_path)
    result = []
    latency_all = [0]

    def get_input_shape(m, in_f, out_f):
        input_shape = list(in_f[0].shape)
        found = False
        for ops in ops_list:
            if isinstance(m, ops[0]) and input_shape == ops[1] and m.__repr__() == ops[-1]:
                result.append([ops[-1], ops[1], list(out_f.shape), ops[2]])
                latency_all[0] += ops[2]
                found = True
                break

        if not found:
            logger.warning("No matching operator is found, check whether the input format of the defined operator is "
                           "consistent with that in the operator description file")
            logger.warning("ops: %s \n ops_list:" % m.__repr__())
            for ops in ops_list:
                print(ops[-1])

    def search_ops(module, ops_list):
        for name, m in module.named_children():
            found = False
            for ops in ops_list:
                if isinstance(m, ops[0]):
                    if verbose:
                        print("{} belongs to {}".format(name, ops[0].__name__))
                    m.register_forward_hook(get_input_shape)
                    m.__name__ = name
                    found = True
                    break
            if not found:
                search_ops(m, ops_list)
        return module

    module = search_ops(module, ops_list)
    # Move model to GPU
    # If test model on cpu may get the "RuntimeError: std::bad_alloc"
    module = module.cuda()
    module.eval()
    x = torch.ones(input_shape).cuda()
    if check_emb(module):
        x = x.long()
    with torch.no_grad():
        _ = module(x)
    result.append(("Sum", "-", "-", latency_all[0]))
    if verbose:
        print(tabulate(result, headers=headers))
    return latency_all[0]


def parse_kargs(module, param):
    ops_register = []
    if param.get("no_params", False):
        for in_shape in param["input_shape"]:
            ops_register.append(ops_meta(module, in_shape))
        return ops_register

    cfg_list = generate_param_list(param)
    for cfg in cfg_list:
        input_shape = cfg["input_shape"]
        del cfg["input_shape"]
        ops_register.append(ops_meta(module, input_shape, init_param=cfg))
    return ops_register


def parse_yaml(path):
    """
    Parse the YAML file and return all operators(a list of class)
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
            if param.get("no_params", False):
                for in_shape in param["input_shape"]:
                    ops_register.append(ops_meta(getattr(pkg, ops), in_shape))
                continue
            cfg_list = generate_param_list(param)
            for cfg in cfg_list:
                input_shape = cfg["input_shape"]
                del cfg["input_shape"]
                ops_register.append(ops_meta(getattr(pkg, ops), input_shape, init_param=cfg))
    return ops_register


def summary_model(model, input_size, ops_list, verbose=False):
    ops_input_shape = []
    headers = ["ops", "input_shape", "out_shape"]

    def get_input_shape(m, in_f, out_f):
        for ops in ops_list:
            if isinstance(m, ops):
                ops_input_shape.append([m.__repr__(), list(in_f[0].shape), list(out_f.shape)])
                break

    def search_ops(module, ops_list):
        for name, m in module.named_children():
            found = False
            for ops in ops_list:
                if isinstance(m, ops):
                    if verbose:
                        print("{} belongs to {}".format(name, ops.__name__))
                    m.register_forward_hook(get_input_shape)
                    m.__name__ = name
                    found = True
                    break
            if not found:
                search_ops(m, ops_list)
        return module

    module = search_ops(model, ops_list)
    device = get_net_device(module)
    module.eval()
    x = torch.ones(input_size).to(device)
    if check_emb(module):
        x = x.long()
    _ = module(x)
    print(tabulate(ops_input_shape, headers=headers))


def check_emb(model):
    is_emb = False
    for m in model.modules():
        is_emb = isinstance(m, torch.nn.Embedding)
        if is_emb:
            return is_emb
    return is_emb
