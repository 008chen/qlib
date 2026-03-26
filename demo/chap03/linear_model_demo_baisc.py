#!/usr/bin/env python3
"""
线性模型演示
基于第3章监督学习模型内容
"""

import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
from qlib.data import D

import logging


def calculate_ma(data, window=20):
    return data.rolling(window=window).mean()
if __name__ == '__main__':
    # 初始化Qlib
    print("初始化Qlib...")
    qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn",logging_level=logging.DEBUG)
    

    stockpool=['sz300840']
    # # 准备数据处理器
    # handler = Alpha158(
    #     instruments=stockpool,
    #     start_time='2023-01-01',
    #     end_time='2026-03-20',
    #     freq='day'
    # )
    
    # 创建数据集
    print("创建数据集...")
    # dataset = DatasetH(
    #     handler=handler,
    #     segments={
    #         'train': ('2023-01-01', '2025-01-01'),
    #         'test': ('2025-01-02', '2026-01-01')
    #     }
    # )

    
    data = D.features(
        instruments=stockpool,
        fields=["$close", "$volume", "$open"],
        start_time="2023-01-01",
        end_time="2026-03-20",
        freq="day"
    )
    # 准备训练数据
    print("准备数据...")
    data['MA5'] = calculate_ma(data['$close'], 5)
    
    # 创建标签（预测未来1天的收益率）
    data['LABEL0'] = data['$close'].shift(-1) / data['$close'] - 1
    
    train_end = '2025-01-01'
    test_start = '2025-01-02'
    # train_data = data.prepare('train')
    # test_data = data.prepare('test')
    
    train_data = data.loc[data.index.get_level_values('datetime') <= train_end]
    test_data = data.loc[data.index.get_level_values('datetime') >= test_start]
    
    # 分离特征和标签
    X_train = train_data.drop('LABEL0', axis=1)
    y_train = train_data['LABEL0']
    X_test = test_data.drop('LABEL0', axis=1)
    y_test = test_data['LABEL0']
    
    # 处理NaN值 - 用均值填充
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())  # 用训练集均值填充测试集
    y_train = y_train.fillna(y_train.mean())
    y_test = y_test.fillna(y_test.mean())
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    print("\n=== 线性回归模型演示 ===")
    
    # 1. 基础线性回归
    print("\n1. 基础线性回归:")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    
    linear_mse = np.mean((linear_pred - y_test.values) ** 2)
    linear_ic = np.corrcoef(linear_pred, y_test.values)[0, 1]
    
    print(f"线性回归 MSE: {linear_mse:.6f}")
    print(f"线性回归 IC: {linear_ic:.4f}")
    
    # 2. Ridge回归（L2正则化）
    print("\n2. Ridge回归 (L2正则化):")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    
    ridge_mse = np.mean((ridge_pred - y_test.values) ** 2)
    ridge_ic = np.corrcoef(ridge_pred, y_test.values)[0, 1]
    
    print(f"Ridge回归 MSE: {ridge_mse:.6f}")
    print(f"Ridge回归 IC: {ridge_ic:.4f}")
    
    # 3. Lasso回归（L1正则化）
    print("\n3. Lasso回归 (L1正则化):")
    lasso_model = Lasso(alpha=0.1, max_iter=1000)
    lasso_model.fit(X_train, y_train)
    lasso_pred = lasso_model.predict(X_test)
    
    lasso_mse = np.mean((lasso_pred - y_test.values) ** 2)
    lasso_ic = np.corrcoef(lasso_pred, y_test.values)[0, 1]
    
    print(f"Lasso回归 MSE: {lasso_mse:.6f}")
    print(f"Lasso回归 IC: {lasso_ic:.4f}")
    
    # 特征选择分析（Lasso的特性）
    non_zero_features = np.sum(lasso_model.coef_ != 0)
    print(f"Lasso选择的非零特征数量: {non_zero_features}/{len(lasso_model.coef_)}")
    
    print("\n=== 模型对比总结 ===")
    print("模型类型\t\tMSE\t\tIC")
    print("-" * 40)
    print(f"线性回归\t\t{linear_mse:.6f}\t{linear_ic:.4f}")
    print(f"Ridge回归\t\t{ridge_mse:.6f}\t{ridge_ic:.4f}")
    print(f"Lasso回归\t\t{lasso_mse:.6f}\t{lasso_ic:.4f}")
    
    print("\n线性模型演示完成！")