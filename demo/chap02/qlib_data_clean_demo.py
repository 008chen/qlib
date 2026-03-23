import qlib
from qlib.data import D
import pandas as pd
import numpy as np

def clean_data(data):
    # 处理缺失值
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    # 处理价格异常值
    price_cols = ['$open', '$high', '$low', '$close']
    for col in price_cols:
        if col in data.columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            data[col] = data[col].clip(lower=mean_val - 3 * std_val, upper=mean_val + 3 * std_val)
    # 成交量异常值处理
    if '$volume' in data.columns:
        data['$volume'] = data['$volume'].clip(lower=0)
    return data

# 初始化Qlib（请根据实际数据路径修改）
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

# 获取特征数据
data = D.features(
    instruments=['SH600000'],
    fields=['$close', '$open', '$high', '$low', '$volume'],
    start_time='2020-01-01',
    end_time='2020-01-31',
    freq='day'
)
print("原始数据:")
print(data.head())

cleaned = clean_data(data)
print("清洗后数据:")
print(cleaned.head()) 