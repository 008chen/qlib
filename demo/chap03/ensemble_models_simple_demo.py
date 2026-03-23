#!/usr/bin/env python3
"""
简化版集成模型示例
基于第3章集成方法内容
"""

import qlib
import numpy as np
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.catboost_model import CatBoostModel
from qlib.contrib.model.xgboost import XGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

def main():
    # 初始化Qlib
    print("初始化Qlib...")
    qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")
    
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
    
    print("创建基础模型...")
    
    # 创建基础模型
    lgb_model = LGBModel(
        loss='mse',
        n_estimators=50,  # 减少迭代次数以节省时间
        max_depth=6,
        learning_rate=0.1,
        verbose=-1
    )
    
    xgb_model = XGBModel(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=50,  # 减少迭代次数
        objective='reg:squarederror',
        random_state=42
    )
    
    cat_model = CatBoostModel(
        iterations=50,  # 减少迭代次数
        learning_rate=0.1,
        depth=6,
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
    
    print("预测结果样例：")
    print("LightGBM预测样例：", lgb_pred.values.flatten()[:5])
    print("XGBoost预测样例：", xgb_pred.values.flatten()[:5])
    print("CatBoost预测样例：", cat_pred.values.flatten()[:5])
    
    # 简单平均集成
    print("\n简单平均集成...")
    avg_pred = (lgb_pred.values.flatten() + 
                xgb_pred.values.flatten() + 
                cat_pred.values.flatten()) / 3
                
    print("平均集成预测样例：", avg_pred[:5])
    
    # 加权平均集成（简单权重）
    print("\n加权平均集成...")
    # 假设权重：LGB=0.4, XGB=0.35, CAT=0.25
    weights = [0.4, 0.35, 0.25]
    weighted_pred = (lgb_pred.values.flatten() * weights[0] + 
                     xgb_pred.values.flatten() * weights[1] + 
                     cat_pred.values.flatten() * weights[2])
                     
    print("加权集成预测样例：", weighted_pred[:5])
    print(f"权重分配 - LGB: {weights[0]}, XGB: {weights[1]}, CAT: {weights[2]}")
    
    print("\n=== 集成模型演示完成 ===")
    print("成功训练了3个基础模型并进行了集成预测")

if __name__ == "__main__":
    main()