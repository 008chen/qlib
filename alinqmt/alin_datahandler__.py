import qlib
import pandas as pd
import talib
import numpy as np
from qlib.data.dataset.loader import NestedDataLoader, StaticDataLoader, QlibDataLoader
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
from qlib.data import D


# 计算 TA-Lib 指标
def calc_talib_features(group):
    
    group = group.dropna()
    
    # print(group)
    # print(type(group))
    close = group['$close'].values.astype(np.float64)
    high = group['$high'].values.astype(np.float64)
    low = group['$low'].values.astype(np.float64)
    
    # 创建特征 DataFrame
    features = pd.DataFrame(index=group.index)
    # features['RSI14'] = talib.RSI(close, timeperiod=14)
    # features['MACD'] = talib.MACD(close, 12, 26, 9)[0]
    # features['BB_upper'] = talib.BBANDS(close, 20)[0]
    features['ATR14'] = talib.ATR(high, low, close, 14)
    # features['SMA20'] = talib.SMA(close, 20)
    # 使用 option_context 临时设置

    return features

if __name__ == "__main__":
    # xx()

    # 1. 初始化 Qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')


    stock_list=["SZ002285","SH601577"]
    # stock_list=["SZ002285"]
    
    start_time = '2020-01-01'
    end_time='2026-02-28'
    # 2. 首先计算 TA-Lib 指标并保存为 StaticDataLoader 可用的格式
    raw_data = D.features(
        instruments=stock_list,
        fields=['$close', '$high', '$low', '$open'], 
        start_time=start_time, 
        end_time=end_time
    )
    # print(raw_data)




    # 分组计算并合并
    talib_features = raw_data.groupby(level='instrument').apply(calc_talib_features)
    talib_features = talib_features.dropna()
    
   

    # 确保列是 MultiIndex 格式 (feature, colname)
    talib_features.columns = pd.MultiIndex.from_product([['feature'], talib_features.columns])
    
    
    alpha158_dl = Alpha158(
        start_time=start_time,  # 根据你的需求调整
        end_time=end_time,
        instruments=stock_list,       # 或其他股票池
        freq="day",
    )
    static_dl = StaticDataLoader(config={"feature": talib_features})

    # 3. 使用 NestedDataLoader 合并 Alpha158 和 TA-Lib 数据
    # nested_loader = NestedDataLoader(
    #     dataloader_l=[
    #         {
    #             "class": "qlib.contrib.data.loader.Alpha158DL",  # Alpha158 数据加载器
    #             "kwargs": {
    #                 "config": {
    #                     "feature": alpha158_handler.get_feature_config(),  # Alpha158 特征
    #                     "label": (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])
    #                 }
    #             }
    #         },
    #         {
    #             "class": "StaticDataLoader",  # TA-Lib 特征
    #             "kwargs": {
    #                 "config": {"feature": talib_features}  # 直接传入 DataFrame
    #             }
    #         }
    #     ],
    #     join="inner"  # 内连接：只保留两个数据源都有的日期
    # )
    nested_loader = NestedDataLoader(
        dataloader_l=[alpha158_dl, static_dl],
        join="inner"
    )

    # 4. 创建 DatasetH
    segments = {
        "train": ("2020-01-01", "2022-12-31"),
        "valid": ("2023-01-01", "2023-06-30"),
        "test": ("2023-07-01", "2024-12-31")
    }

    dataset = DatasetH(
        handler={"class": "DataHandlerLP", "kwargs": {"data_loader": nested_loader}},
        segments=segments
    )

    # 5. 准备数据并训练
    x_train, y_train = dataset.prepare("train", col_set=["feature", "label"])
    print(f"合并后特征维度: {x_train.shape}")  # 包含 Alpha158 + TA-Lib 特征

    # 6. 训练模型
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