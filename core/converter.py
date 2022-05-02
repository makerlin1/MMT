# _*_ coding: utf-8 _*_
"""
Time:     2022-05-02 0:26
Author:   Haolin Yan(XiDian University)
File:     converter.py
"""
from .utils import *
from .parser import parse_yaml
import logging
import os
import pickle
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert2mnn(model, input_shape, path, verbose=False):
    """Convert pytorch model to mnn format."""
    model.eval().cuda()
    dummy_input = torch.randn(input_shape, device='cuda')
    input_names = ["input"]
    output_names = ["output"]
    assert isinstance(input_shape, list), "Invalid input_shape: {}".format(input_shape)
    name = os.path.join(path, replace(model.__repr__())) + shape2str(input_shape)
    onnx_name = name + ".onnx"
    mnn_name = name + ".mnn"
    try:
        torch.onnx.export(model, dummy_input, onnx_name, verbose=verbose, input_names=input_names,
                      output_names=output_names)
    except:
        logger.warning("%s maybe is a invalid ops!" % (onnx_name))
        return 0
    logger.info("Export %s to %s" % (name, onnx_name))
    cmd = "MNNConvert -f ONNX --modelFile {} --MNNModel {} --bizCode biz".format(onnx_name, mnn_name)
    os.system(cmd)
    if not os.path.isfile(mnn_name):
        raise ValueError("Fail to convert %s to %s" % (onnx_name, mnn_name))
    logger.info("Convert %s to %s" % (onnx_name, mnn_name))
    os.remove(onnx_name)
    return os.path.basename(mnn_name)


def generate_ops_list(path, fp):
    """
    Generate operators in mnn format according to the specified YAML file.
    """
    operator_list = parse_yaml(path)
    logger.info("Begin to convert %d ops!" % (len(operator_list)))
    if not os.path.isdir(fp):
        os.mkdir(fp)
        logger.info("Create %s" % fp)
    success_ops = []
    for i, ops in enumerate(operator_list):
        module, input_shape = ops.return_instance(), ops.input_shape
        mnn_name = convert2mnn(module, input_shape, fp, verbose=False)
        if mnn_name != 0:
            ops.record_mnn_fname(mnn_name)
            success_ops.append(ops)
    logger.info("Conversion completed (%d successes, %d failures)" % (len(success_ops), len(operator_list) - len(success_ops)))
    meta_fpath = os.path.join(fp, "meta.pkl")
    with open(meta_fpath, "wb") as f:
        pickle.dump(success_ops, f)
