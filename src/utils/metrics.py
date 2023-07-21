import numpy as np

def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))

def get_RMSE(pred, real):
    return np.sqrt(get_MSE(pred, real))

def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))

def get_SMAPE(pred, real):
    numerator = np.abs(real - pred)
    denominator = (np.abs(real) + np.abs(pred))
    result = numerator / denominator
    result[np.isnan(result)] = 1e10
    return np.mean(result * (numerator > 0))

def get_MAPE(pred, real):
    ori_real = real.copy()
    epsilon = 1 # if use small number like 1e-5 resulting in very large value
    real[real == 0] = epsilon
    return np.mean(np.abs(ori_real - pred) / real)

def get_all_metrics(pred, real):
    mae = get_MAE(pred, real)
    rmse = get_RMSE(pred, real)
    smape = get_SMAPE(pred, real)
    mape = get_MAPE(pred, real)
    return mae, rmse, mape, smape