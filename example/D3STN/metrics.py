# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Some useful metrics
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import numpy as np


def MSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        # std scaler inverse_transform maybe change 0 to 0.0001
        mask = np.greater_equal(y_true, 0.1)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(np.subtract(y_pred, y_true))
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse


def RMSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        # std scaler inverse_transform maybe change 0 to 0.0001
        mask = np.greater_equal(y_true, 0.1)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(np.subtract(y_pred, y_true)))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        # std scaler inverse_transform maybe change 0 to 0.0001
        mask = np.greater_equal(y_true, 0.1)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(y_pred, y_true))
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def MAPE(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            # std scaler inverse_transform maybe change 0 to 0.0001
            mask = np.greater_equal(y_true, null_val)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100
