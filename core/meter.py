# _*_ coding: utf-8 _*_
"""
Time:     2022-05-02 13:54
Author:   Haolin Yan(XiDian University)
File:     meter.py
"""
import os
import re
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_latency(path, times=30):
    cmd = "MNNV2Basic.out %s %d 0 0 4 > %s" % (path, times, path + '.log')
    os.system(cmd)
    with open(path + '.log', 'r') as f:
        lines = f.readlines()
    logger.info("%s:::%s" % (path, lines[-1]))
    prefix = ["Avg", "Min", "Max"]
    result = dict((prefix[i], float(v)) for i, v in enumerate(re.findall(r"\d+\.?\d*", lines[-1])))
    result["repeat"] = times
    return result


def meter_ops(fp, times=30):
    logger.info("Begin to measure!")
    meta_path = os.path.join(fp, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta_list = pickle.load(f)
    for meta in meta_list:
        path = os.path.join(fp, meta.mnn_fname)
        result = get_latency(path, times=times)
        meta.record_mnn_performance(result)
    update_meta_path = os.path.join(fp, "meta_lantency.pkl")
    with open(update_meta_path, "wb") as f:
        pickle.dump(meta_list, f)
    logger.info("Finish!")

