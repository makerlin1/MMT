# _*_ coding: utf-8 _*_
"""
Time:     2022-05-02 13:54
Author:   Haolin Yan(XiDian University)
File:     meter.py
"""
import json
import os
import re
import pickle
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_latency(path, times=30, verbose=True):
    """Test the latency of operator with the given path"""
    cmd = "MNNV2Basic.out %s %d 0 0 4 > %s" % (path, times, path + '.log')
    os.system(cmd)
    with open(path + '.log', 'r') as f:
        lines = f.readlines()
    if verbose:
        print(lines)
    logger.info("%s:::%s" % (path, lines[-1]))
    prefix = ["Avg", "Min", "Max"]
    result = dict((prefix[i], float(v)) for i, v in enumerate(re.findall(r"\d+\.?\d*", lines[-1])))
    result["repeat"] = times
    return result


def meter_ops(fp, times=30, verbose=False):
    """Test the latency of operators in the input folder, and write the result in meta_latency.pkl"""
    logger.info("Begin to measure!")
    meta_path = os.path.join(fp, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta_list = pickle.load(f)
    for meta in meta_list:
        path = os.path.join(fp, meta.mnn_fname)
        result = get_latency(path, times=times, verbose=verbose)
        meta.record_mnn_performance(result)
    update_meta_path = os.path.join(fp, "meta_latency.pkl")
    with open(update_meta_path, "wb") as f:
        pickle.dump(meta_list, f)
    logger.info("Finish!")



def get_model_latency(model, input_shape, path=".", times=30):
    """
    Convert pytorch model to mnn format and test its latency
    """
    if len(model.__repr__()) > 50:
        logger.warn("please rewrite the model.__repr__()")
    from .converter import convert2mnn
    mnn_name = convert2mnn(model, input_shape, path, verbose=False)
    path = os.path.join(path, mnn_name)
    result = get_latency(path, times=times)
    os.remove(path)
    return result


def meter_models(fp, times=30, verbose=False):
    """Test the latency of model in the input folder, and write the result in *meta.json"""
    logger.info("Begin to measure!")
    fnames = [x for x in os.listdir(fp) if x[-3:] == "mnn"]
    for name in fnames:
        fpath = os.path.join(fp, name)
        result = get_latency(fpath, times=times, verbose=verbose)
        json_path = os.path.join(fp, name[:-4] + "meta.json")
        with open(json_path, "r") as f:
            kargs = json.load(f)
        kargs["latency"] = result
        with open(json_path, "w") as f:
            json.dump(kargs, f)
    logger.info("Finish!")