import qlib
from qlib.data import D
import pandas as pd
import numpy as np

def calculate_ma(data, window=20):
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20, num_std=2):
    ma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    return upper_band, ma, lower_band

# 初始化Qlib（请根据实际数据路径修改）
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

# 获取特征数据
data = D.features(
    instruments=['SH600000'],
    fields=['$close'],
    start_time='2020-01-01',
    end_time='2020-03-31',
    freq='day'
)

# 计算移动平均线
data['MA5'] = calculate_ma(data['$close'], 5)
data['MA20'] = calculate_ma(data['$close'], 20)

# 计算RSI
data['RSI'] = calculate_rsi(data['$close'])

# 计算布林带
upper, middle, lower = calculate_bollinger_bands(data['$close'])
data['BB_upper'] = upper
data['BB_middle'] = middle
data['BB_lower'] = lower

print(data.tail(10)) 