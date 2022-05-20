# _*_ coding: utf-8 _*_
"""
Time:     2022-05-05 17:08
Author:   Haolin Yan(XiDian University)
File:     predictor.py
"""
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from .parser import predict_latency
import numpy as np


class latency_predictor:
    def __init__(self, ops_table, data):
        self.data = pd.read_csv(data)
        X = self.data["latency_true(ms)"].values
        y = self.data["latency_pred(ms)"].values
        # self.gp = get_gp(X[:, None], y[:, None])
        self.gp = get_gp(y[:, None], X[:, None])

    def __call__(self, model, ops_path, input_shape, verbose=False):
        latency = np.array([predict_latency(model, ops_path, input_shape, verbose=verbose)])
        y = self.gp.predict(latency[:, None])
        return y[0][0]


def get_gp(X, Y):
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(X, Y)
    return gpr


def get_linear(X, Y):
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    linreg.fit(X, Y)
    return linreg
