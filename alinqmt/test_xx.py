import qlib
import pandas as pd
import talib
import numpy as np
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
from qlib.data import D

import alin_datahandler 

stock_list=["SZ002285","SH601577"]
start_time = '2020-01-01'
end_time='2026-02-28'
    
if __name__ == "__main__":
    # 1. 初始化 Qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')

    # 2. 获取 Alpha158 数据
    alpha158_handler = alin_datahandler.CustomAlpha158(
        instruments=stock_list,
        start_time=start_time,
        end_time=end_time,
        freq="day",
        infer_processors=[],  # 可自定义预处理器
        learn_processors=[],  # 可自定义标签处理器
    )
    

    # 获取 Alpha158 的原始数据（包含特征和标签）
    alpha158_data = alpha158_handler.fetch(data_key=DataHandlerLP.DK_L)

    # 3. 计算 TA-Lib 指标
    # instruments = D.instruments(market='csi300')
    raw_data = D.features(
        instruments=stock_list, 
        fields=['$close', '$high', '$low', '$open', '$volume'], 
        start_time=start_time, 
        end_time=end_time
    )
 

    def calc_talib_features(group):
        group = group.dropna()
        close = group['$close'].values.astype(np.float64)
        high = group['$high'].values.astype(np.float64)
        low = group['$low'].values.astype(np.float64)
        
        features = pd.DataFrame(index=group.index)
        # features['RSI14'] = talib.RSI(close, 14)
        # features['MACD'] = talib.MACD(close, 12, 26, 9)[0]
        # features['BB_upper'] = talib.BBANDS(close, 20)[0]
        features['ATR14'] = talib.ATR(high, low, close, 14)
        # features['SMA20'] = talib.SMA(close, 20)
        return features
    
    # print(raw_data)
    talib_features = raw_data.groupby(level='instrument').apply(calc_talib_features)
    talib_features = talib_features.droplevel(0)
    talib_features = talib_features.reset_index().set_index(['datetime', 'instrument']).sort_index()
    
    # talib_features = talib_features.dropna()
    # print(talib_features)
    # print(raw_data)
    
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # print(talib_features)
    # print("alpha158_data")
    # print(alpha158_data)

    # 5. 使用 DataHandlerLP.from_df() 创建 Handler
    # 需要将数据转换为正确的 MultiIndex 格式
    # 假设 alpha158_data 已经有 (feature, col) 和 (label, col) 的列结构

    # 如果 talib_features 没有 MultiIndex，需要转换


    
    
    # 重新合并
    combined_data = alpha158_data.join(talib_features, how='inner')
    print(combined_data)
    
    new_columns = []
    label_cols = ['LABEL0']
    for col in combined_data.columns.get_level_values(0):
        if col not in label_cols:
            new_columns.append(('feature', col))
        else:
            new_columns.append(('label', col))

    combined_data.columns = pd.MultiIndex.from_tuples(new_columns)
    
    # if not isinstance(combined_data.columns, pd.MultiIndex):
    #     combined_data.columns = pd.MultiIndex.from_product([['feature'], talib_features.columns])
    # print(combined_data)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(combined_data)

    # 创建 Handler
    handler = DataHandlerLP.from_df(combined_data)

    # 6. 创建 DatasetH 并训练
    segments = {
        "train": ("2020-01-01", "2022-12-31"),
        "valid": ("2023-01-01", "2023-06-30"),
        "test": ("2023-07-01", "2024-12-31")
    }

    dataset = DatasetH(handler=handler, segments=segments)

    # 7. 训练模型
    model = LGBModel(
        objective='regression',
        metric='mse',
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1
    )

    model.fit(dataset)

    # 8. 预测
    pred = model.predict(dataset)
    print(f"预测结果: {pred.head()}")