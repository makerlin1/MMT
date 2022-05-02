# _*_ coding: utf-8 _*_
"""
Time:     2022-05-02 13:20
Author:   Haolin Yan(XiDian University)
File:     parser.py
"""
from .utils import import_module, generate_param_list, ops_meta
import yaml


# todo: 解析模型不仅需要解析类名同时还有输入的张量尺寸
def parser_model(module, ops_list, verbose=False):
    """
    Parse the operators in the model and return the corresponding dictionary
    e.g.
        ops_list = [MLP, nn.Linear, nn.ReLU]  # MLP is designed module by yourself
        arch2ops = parser_model(model, ops_list, verbose=True)
    """
    arch2ops = {}

    def get_input_shape(ops, input):
        pass

    def search_ops(module, ops_list):
        for name, m in module.named_children():
            found = False
            for ops in ops_list:
                if isinstance(m, ops):
                    if verbose:
                        print("{} belongs to {}".format(name, ops.__name__))
                    found = True
                    arch2ops[name] = ops.__name__
                    m.__name__ = name
                    m.register_forward_hook(get_input_shape)
                    break
            if not found:
                search_ops(m, ops_list)

    search_ops(module, ops_list)
    return arch2ops


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
