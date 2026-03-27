#!/usr/bin/env python3
"""
模型训练流程演示
基于第4章模型训练与优化内容
"""

import qlib
import numpy as np
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

def time_series_split(data, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2):
    """时间序列数据分割"""
    total_length = len(data)
    train_end = int(total_length * train_ratio)
    valid_end = int(total_length * (train_ratio + valid_ratio))
    
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    
    return train_data, valid_data, test_data




# 假设 data 长度为 500，window_size=100，step_size=50：
# 第一轮 (i=0):
# 训练集: 索引 0 到 100
# 验证集: 索引 100 到 150
# 第二轮 (i=50):
# 训练集: 索引 50 到 150 (窗口向前滚动了50)
# 验证集: 索引 150 到 200
# 第三轮 (i=100):
# 训练集: 索引 100 到 200
# 验证集: 索引 200 到 250
def rolling_window_split(data, window_size=252, step_size=63):
    """滚动窗口分割"""
    splits = []
    
    for i in range(0, len(data) - window_size, step_size):
        # 1. 确定训练集的起止位置
        train_start = i
        train_end = i + window_size
        
        # 2. 确定验证集的结束位置
        # 验证集紧接在训练集之后，长度通常为 step_size
        # 使用 min 函数是为了防止索引越界（处理数据末尾不足一个步长的情况）
        valid_end = min(train_end + step_size, len(data))
        
        
        # 安全检查：如果验证集结束位置没有超过训练集结束位置，
        # 说明剩余数据不足以构成有效的验证集，直接跳出循环
        if valid_end <= train_end:
            break
            
        train_data = data[train_start:train_end]
        valid_data = data[train_end:valid_end]
        
        splits.append((train_data, valid_data))
    
    return splits

def calculate_ic(predictions, returns):
    """计算IC值"""
    # 移除缺失值
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_clean = predictions[valid_mask]
    ret_clean = returns[valid_mask]
    
    if len(pred_clean) <= 1:
        return np.nan
    
    # 计算相关系数
    correlation = np.corrcoef(pred_clean, ret_clean)[0, 1]
    
    return correlation

def main():
    print("准备数据...")
    
    # 准备数据处理器
    handler = Alpha158(
        instruments='csi300',
        start_time='2020-01-01',
        end_time='2020-12-31',
        fit_start_time='2020-01-01',
        fit_end_time='2020-06-30',
        freq='day'
    )
    
    # 使用DatasetH管理数据分割
    from qlib.data.dataset import DatasetH
    dataset = DatasetH(
        handler=handler,
        segments={
            'train': ('2020-01-01', '2020-06-30'),
            'valid': ('2020-07-01', '2020-09-30'),
            'test': ('2020-10-01', '2020-12-31')
        }
    )
    
    # 获取数据用于展示
    train_data = dataset.prepare('train')
    valid_data = dataset.prepare('valid')
    test_data = dataset.prepare('test')
    
    # 为了兼容后续代码，创建data字典
    data = {
        'train': {'feature': train_data.drop('LABEL0', axis=1), 'label': train_data[['LABEL0']]},
        'valid': {'feature': valid_data.drop('LABEL0', axis=1), 'label': valid_data[['LABEL0']]},
        'test': {'feature': test_data.drop('LABEL0', axis=1), 'label': test_data[['LABEL0']]}
    }
    
    print("=== 数据分割演示 ===")
    
    # 1. 基础时间序列分割
    print("1. 基础时间序列分割")
    full_data = pd.concat([
        data['train']['feature'].join(data['train']['label']),
        data['valid']['feature'].join(data['valid']['label']),
        data['test']['feature'].join(data['test']['label'])
    ])
    
    train_split, valid_split, test_split = time_series_split(full_data)
    print(f"训练集大小: {len(train_split)}")
    print(f"验证集大小: {len(valid_split)}")  
    print(f"测试集大小: {len(test_split)}")
    
    # 2. 滚动窗口分割演示
    print("\n2. 滚动窗口分割演示")
    splits = rolling_window_split(full_data, window_size=100, step_size=30)
    print(f"生成了 {len(splits)} 个滚动窗口")
    
    print("\n=== 模型训练演示 ===")
    
    # 基础模型训练
    print("训练基础LightGBM模型...")
    model = LGBModel(
        loss='mse',
        colsample_bytree=0.8,
        learning_rate=0.1,
        subsample=0.8,
        n_estimators=100,
        max_depth=6,
        num_leaves=64,
        min_child_samples=20,
        verbose=-1
    )
    
    # 训练模型
    model.fit(dataset)
    
    # 预测
    train_pred = model.predict(dataset, segment='train')
    valid_pred = model.predict(dataset, segment='valid')
    test_pred = model.predict(dataset, segment='test')
    
    print("\n=== 模型评估 ===")
    
    # 计算IC指标
    train_ic = calculate_ic(train_pred, data['train']['label']['LABEL0'].values)
    valid_ic = calculate_ic(valid_pred, data['valid']['label']['LABEL0'].values)
    test_ic = calculate_ic(test_pred, data['test']['label']['LABEL0'].values)
    
    print(f"训练集IC: {train_ic:.4f}")
    print(f"验证集IC: {valid_ic:.4f}")
    print(f"测试集IC: {test_ic:.4f}")
    
    # 计算MSE
    train_mse = mean_squared_error(data['train']['label']['LABEL0'], train_pred)
    valid_mse = mean_squared_error(data['valid']['label']['LABEL0'], valid_pred)
    test_mse = mean_squared_error(data['test']['label']['LABEL0'], test_pred)
    
    print(f"训练集MSE: {train_mse:.6f}")
    print(f"验证集MSE: {valid_mse:.6f}")
    print(f"测试集MSE: {test_mse:.6f}")
    
    print("\n=== 时间序列交叉验证 ===")
    
    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    X = data['train']['feature']
    y = data['train']['label']['LABEL0']
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"处理第 {fold + 1} 折...")
        
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # 训练折叠模型
        fold_model = LGBModel(
            loss='mse',
            n_estimators=50,
            max_depth=6,
            verbose=-1
        )
        
        fold_model.fit(X_train_fold, y_train_fold)
        fold_pred = fold_model.predict(X_val_fold)
        
        # 计算IC
        fold_ic = calculate_ic(fold_pred, y_val_fold.values)
        cv_scores.append(fold_ic)
        
        print(f"第 {fold + 1} 折 IC: {fold_ic:.4f}")
    
    print(f"交叉验证平均IC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    print("\n=== 滚动窗口验证 ===")
    
    # 滚动窗口验证
    rolling_scores = []
    
    for i, (train_window, val_window) in enumerate(splits[:3]):  # 只测试前3个窗口
        print(f"处理滚动窗口 {i + 1}...")
        
        # 分离特征和标签
        train_features = train_window.drop(['LABEL0'], axis=1)
        train_labels = train_window['LABEL0']
        val_features = val_window.drop(['LABEL0'], axis=1)
        val_labels = val_window['LABEL0']
        
        # 训练模型
        rolling_model = LGBModel(
            loss='mse',
            n_estimators=50,
            max_depth=6,
            verbose=-1
        )
        
        rolling_model.fit(train_features, train_labels)
        rolling_pred = rolling_model.predict(val_features)
        
        # 计算IC
        rolling_ic = calculate_ic(rolling_pred, val_labels.values)
        rolling_scores.append(rolling_ic)
        
        print(f"滚动窗口 {i + 1} IC: {rolling_ic:.4f}")
    
    print(f"滚动窗口平均IC: {np.mean(rolling_scores):.4f} ± {np.std(rolling_scores):.4f}")
    
    print("\n训练流程演示完成！")

if __name__ == "__main__":
    main()