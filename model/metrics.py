import numpy as np

def cal_mape(y, y_pred):
    #print(y, y_pred)
    y, y_pred = np.array(y), np.array(y_pred)
    mape = np.mean(np.abs(y - y_pred) / (y+1))
    mape = round(mape, 2)
    return mape