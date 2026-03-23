import qlib
from qlib.data import D
import pandas as pd
import numpy as np

def check_completeness(data):
    missing_stats = data.isnull().sum()
    missing_ratio = missing_stats / len(data)
    print("缺失值统计:")
    for col, ratio in missing_ratio.items():
        if ratio > 0:
            print(f"{col}: {ratio:.2%}")
    return missing_ratio

def check_consistency(data):
    if all(col in data.columns for col in ['$open', '$high', '$low', '$close']):
        invalid_high = data['$high'] < data[['$open', '$close']].max(axis=1)
        invalid_low = data['$low'] > data[['$open', '$close']].min(axis=1)
        print(f"高价逻辑错误: {invalid_high.sum()} 条")
        print(f"低价逻辑错误: {invalid_low.sum()} 条")
    if '$volume' in data.columns:
        negative_volume = data['$volume'] < 0
        print(f"负成交量: {negative_volume.sum()} 条")

if __name__ == "__main__":
    # 初始化Qlib（请根据实际数据路径修改）
    qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

    # 获取特征数据
    data = D.features(
        instruments=['SH600000'],
        fields=['$close', '$open', '$high', '$low', '$volume'],
        start_time='2020-01-01',
        end_time='2020-03-31',
        freq='day'
    )

    print("数据完整性检查:")
    check_completeness(data)

    print("数据一致性检查:")
    check_consistency(data) 