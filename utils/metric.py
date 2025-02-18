import numpy as np

def MSE(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))

def MAPE(y_true, y_pred):
    # 避免除以零
    y_true= np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs(y_true- y_pred) / y_true) * 100


def MSPE(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred) / y_true) * 10000

def metric(y_true, y_pred):
    mse = MSE(y_true, y_pred)
    mae = MAE(y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    mspe = MSPE(y_true,y_pred)
    return mse, mae, mape, mspe, rmse
