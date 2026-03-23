#!/usr/bin/env python3
"""
集成模型示例
基于第3章集成方法内容
"""

import qlib
import numpy as np
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.catboost_model import CatBoostModel
from qlib.contrib.model.xgboost import XGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

def main():
    print("准备数据...")
    
    # 准备数据
    handler = Alpha158(
        instruments='csi300',
        start_time='2020-01-01',
        end_time='2020-12-31',
        freq='day'
    )
    
    # 创建数据集
    dataset = DatasetH(
        handler=handler,
        segments={
            'train': ('2020-01-01', '2020-06-30'),
            'valid': ('2020-07-01', '2020-09-30'),
            'test': ('2020-10-01', '2020-12-31')
        }
    )
    
    # 准备数据
    train_data = dataset.prepare('train')
    valid_data = dataset.prepare('valid')
    test_data = dataset.prepare('test')
    
    # 分离特征和标签
    X_train = train_data.drop('LABEL0', axis=1)
    y_train = train_data['LABEL0']
    X_valid = valid_data.drop('LABEL0', axis=1)
    y_valid = valid_data['LABEL0']
    X_test = test_data.drop('LABEL0', axis=1)
    y_test = test_data['LABEL0']
    
    print("创建基础模型...")

        
    



    # 创建基础模型
    lgb_model = LGBModel(             
        loss='mse',                  #损失函数
        colsample_bytree=0.8879,     #特征采样比例
        learning_rate=0.1,           #学习率
        subsample=0.8789,            #样本采样比例
        n_estimators=100,            #树的数量
        max_depth=6,                 #最大深度
        num_leaves=64,            
        min_child_samples=20      
    )
    
    xgb_model = XGBModel(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        objective='reg:squarederror',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    cat_model = CatBoostModel(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        random_seed=42
    )
    
    # 训练基础模型
    print("训练LightGBM...")
    lgb_model.fit(dataset)
    
    print("训练XGBoost...")
    xgb_model.fit(dataset)
    
    print("训练CatBoost...")
    cat_model.fit(dataset)
    
    # 单模型预测
    print("单模型预测...")
    lgb_pred = lgb_model.predict(dataset, segment='test')
    xgb_pred = xgb_model.predict(dataset, segment='test')
    cat_pred = cat_model.predict(dataset, segment='test')
    
    # 获取真实标签
    y_true = y_test
    
    # 计算单模型性能 (处理NaN值)
    lgb_pred_clean = lgb_pred.values.flatten()
    xgb_pred_clean = xgb_pred.values.flatten()
    cat_pred_clean = cat_pred.values.flatten()
    y_true_clean = y_true.values
    
    # 找到非NaN的索引
    valid_idx = ~(np.isnan(lgb_pred_clean) | np.isnan(xgb_pred_clean) | np.isnan(cat_pred_clean) | np.isnan(y_true_clean))
    
    if valid_idx.sum() > 0:
        lgb_mse = mean_squared_error(y_true_clean[valid_idx], lgb_pred_clean[valid_idx])
        xgb_mse = mean_squared_error(y_true_clean[valid_idx], xgb_pred_clean[valid_idx])
        cat_mse = mean_squared_error(y_true_clean[valid_idx], cat_pred_clean[valid_idx])
    else:
        lgb_mse = xgb_mse = cat_mse = float('nan')
    
    print(f"LightGBM MSE: {lgb_mse:.6f}")
    print(f"XGBoost MSE: {xgb_mse:.6f}")
    print(f"CatBoost MSE: {cat_mse:.6f}")
    
    # 简单平均集成
    print("简单平均集成...")
    if valid_idx.sum() > 0:
        avg_pred = (lgb_pred_clean + xgb_pred_clean + cat_pred_clean) / 3
        avg_mse = mean_squared_error(y_true_clean[valid_idx], avg_pred[valid_idx])
    else:
        avg_mse = float('nan')
    print(f"平均集成 MSE: {avg_mse:.6f}")
    
    # 加权平均集成（基于验证集性能）
    print("加权平均集成...")
    lgb_val_pred = lgb_model.predict(dataset, segment='valid')
    xgb_val_pred = xgb_model.predict(dataset, segment='valid')
    cat_val_pred = cat_model.predict(dataset, segment='valid')
    
    y_val = y_valid
    
    lgb_val_mse = mean_squared_error(y_val.values, lgb_val_pred.values.flatten())
    xgb_val_mse = mean_squared_error(y_val.values, xgb_val_pred.values.flatten())
    cat_val_mse = mean_squared_error(y_val.values, cat_val_pred.values.flatten())
    
    # 计算权重（MSE越小权重越大）
    mse_scores = np.array([lgb_val_mse, xgb_val_mse, cat_val_mse])
    weights = 1 / mse_scores
    weights = weights / weights.sum()
    
    print(f"权重分配 - LGB: {weights[0]:.3f}, XGB: {weights[1]:.3f}, CAT: {weights[2]:.3f}")
    
    weighted_pred = (lgb_pred.values.flatten() * weights[0] + 
                     xgb_pred.values.flatten() * weights[1] + 
                     cat_pred.values.flatten() * weights[2])
    
    weighted_mse = mean_squared_error(y_true.values, weighted_pred)
    print(f"加权集成 MSE: {weighted_mse:.6f}")
    
    # Stacking集成（简化版本）
    print("Stacking集成...")
    
    # 使用验证集训练meta-model
    val_features = np.column_stack([lgb_val_pred.values.flatten(), xgb_val_pred.values.flatten(), cat_val_pred.values.flatten()])
    
    # 简单线性回归作为meta-model
    from sklearn.linear_model import Ridge
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(val_features, y_val.values)
    
    # 测试集预测
    test_features = np.column_stack([lgb_pred.values.flatten(), xgb_pred.values.flatten(), cat_pred.values.flatten()])
    stacking_pred = meta_model.predict(test_features)
    stacking_mse = mean_squared_error(y_true.values, stacking_pred)
    print(f"Stacking集成 MSE: {stacking_mse:.6f}")
    
    # 结果对比
    print("\n=== 模型性能对比 ===")
    results = {
        'LightGBM': lgb_mse,
        'XGBoost': xgb_mse,
        'CatBoost': cat_mse,
        '简单平均': avg_mse,
        '加权平均': weighted_mse,
        'Stacking': stacking_mse
    }
    
    print(f"{'LightGBM':10s}: MSE={lgb_mse:.6f}")
    print(f"{'XGBoost':10s}: MSE={xgb_mse:.6f}")
    print(f"{'CatBoost':10s}: MSE={cat_mse:.6f}")
    print(f"{'简单平均':10s}: MSE={avg_mse:.6f}")
    print(f"{'加权平均':10s}: MSE={weighted_mse:.6f}")
    print(f"{'Stacking':10s}: MSE={stacking_mse:.6f}")
    
    # 预测结果样例
    print("\n预测结果样例（前10个）：")
    comparison_df = pd.DataFrame({
        '真实值': y_true.values[:10],
        'LightGBM': lgb_pred.values.flatten()[:10],
        'XGBoost': xgb_pred.values.flatten()[:10],
        'CatBoost': cat_pred.values.flatten()[:10],
        '加权集成': weighted_pred[:10]
    })
    print(comparison_df)

if __name__ == "__main__":
    main()