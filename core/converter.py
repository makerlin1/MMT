# _*_ coding: utf-8 _*_
"""
Time:     2022-05-02 0:26
Author:   Haolin Yan(XiDian University)
File:     converter.py
"""
from utils import *
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_ops_list(path, fp):
    operator_list = parse_yaml(path)
    logger.info("Begin to convert %d ops!" % (len(operator_list)))
    if not os.path.isdir(fp):
        os.mkdir(fp)
        logger.info("Create %s" % fp)
    for ops in operator_list:
        convert2mnn(ops, fp, verbose=False)
    logger.info("Conversion completed")







