from flask import abort
import numpy as np
from main import AI
import json
import dateutil.relativedelta
import datetime
from yahoo_fin.stock_info import *
from sklearn.preprocessing import MinMaxScaler

def load_data(ticker):
    date = datetime.datetime.now()
    date = date + dateutil.relativedelta.relativedelta(months=-2)
    date = date.strftime("%d/%m/%Y")
    stock_info = get_data(ticker=ticker, start_date=date)
    stock_info = stock_info[['close']].tail(22)
    global recent_data
    recent_data[:, 0] = stock_info['close'].tolist()
    stock_info.index = stock_info.index.strftime("%d.%m")
    global recent_data_dates
    recent_data_dates[:, 0] = stock_info.index.tolist()

ai = AI()
ticker = "^GSPC"
recent_data = np.zeros((22, 1))
recent_data_dates = np.zeros((22, 1))
load_data(ticker)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def read_data():
    return recent_data.tolist()


def read_dates():
    return recent_data_dates.tolist()


def make_estimation():
    global recent_data
    recent_data_copy = recent_data.copy()
    sc = MinMaxScaler()
    for i in range(5):
        DataScaler = sc.fit(recent_data_copy)
        recent_data_copy_scaled = DataScaler.transform(recent_data_copy)
        recent_data_copy_scaled = np.swapaxes(recent_data_copy_scaled, 0, 1)
        recent_data_copy_scaled = np.reshape(recent_data_copy_scaled, (1, 22, 1))
        prediction_now = ai.regressor.predict(recent_data_copy_scaled)
        prediction_now = prediction_now * (np.max(recent_data_copy) - np.min(recent_data_copy)) + np.min(recent_data_copy)
        for j in range(0, 21, 1):
            recent_data_copy[j, 0] = recent_data_copy[j+1, 0]
        recent_data_copy[21, 0] = prediction_now
    return recent_data_copy[17:22, 0].tolist()

def ticker(data):
    ticker = data.get("ticker")
    load_data(ticker)
    return 200
