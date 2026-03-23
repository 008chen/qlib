#!/usr/bin/env python3
"""
稳定版集成模型示例
避免内存问题和分段错误
基于第3章集成方法内容
"""

import qlib
import numpy as np
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
import gc  # 垃圾回收

def train_single_model(dataset, model_type='lgb'):
    """训练单个模型以避免内存冲突"""
    print(f"训练{model_type}模型...")
    
    if model_type == 'lgb':
        model = LGBModel(
            loss='mse',
            n_estimators=30,  # 减少迭代次数
            max_depth=4,      # 减少深度
            learning_rate=0.1,
            verbose=-1
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 训练模型
    model.fit(dataset)
    
    # 预测
    train_pred = model.predict(dataset, segment='train')
    test_pred = model.predict(dataset, segment='test')
    
    # 清理内存
    gc.collect()
    
    return {
        'model': model,
        'train_pred': train_pred,
        'test_pred': test_pred
    }

def main():
    # 初始化Qlib
    print("初始化Qlib...")
    qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")
    
    print("准备数据...")
    
    # 准备数据
    handler = Alpha158(
        instruments='csi300',
        start_time='2020-01-01',
        end_time='2020-06-30',  # 减少数据量
        freq='day'
    )
    
    # 创建数据集
    dataset = DatasetH(
        handler=handler,
        segments={
            'train': ('2020-01-01', '2020-04-30'),
            'test': ('2020-05-01', '2020-06-30')
        }
    )
    
    # 准备标签数据用于评估
    test_data = dataset.prepare('test')
    y_test = test_data['LABEL0']
    
    print("=== 单模型训练 ===")
    
    # 训练多个LightGBM模型（不同参数）作为基础模型
    models_results = []
    
    # 模型1：较深的树
    print("训练模型1（深树）...")
    model1 = LGBModel(
        loss='mse',
        n_estimators=20,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=32,
        verbose=-1
    )
    model1.fit(dataset)
    pred1 = model1.predict(dataset, segment='test')
    models_results.append(('深树模型', pred1))
    
    # 清理内存
    del model1
    gc.collect()
    
    # 模型2：较浅的树
    print("训练模型2（浅树）...")
    model2 = LGBModel(
        loss='mse',
        n_estimators=30,
        max_depth=3,
        learning_rate=0.15,
        num_leaves=16,
        verbose=-1
    )
    model2.fit(dataset)
    pred2 = model2.predict(dataset, segment='test')
    models_results.append(('浅树模型', pred2))
    
    # 清理内存
    del model2
    gc.collect()
    
    # 模型3：高学习率
    print("训练模型3（高学习率）...")
    model3 = LGBModel(
        loss='mse',
        n_estimators=15,
        max_depth=4,
        learning_rate=0.2,
        num_leaves=24,
        verbose=-1
    )
    model3.fit(dataset)
    pred3 = model3.predict(dataset, segment='test')
    models_results.append(('高学习率模型', pred3))
    
    # 清理内存
    del model3
    gc.collect()
    
    print("\n=== 集成方法演示 ===")
    
    # 提取预测结果
    pred1_vals = pred1.values.flatten()
    pred2_vals = pred2.values.flatten()
    pred3_vals = pred3.values.flatten()
    
    print("单模型预测样例：")
    print(f"模型1前5个预测: {pred1_vals[:5]}")
    print(f"模型2前5个预测: {pred2_vals[:5]}")
    print(f"模型3前5个预测: {pred3_vals[:5]}")
    
    # 1. 简单平均集成
    avg_pred = (pred1_vals + pred2_vals + pred3_vals) / 3
    print(f"\n简单平均集成前5个预测: {avg_pred[:5]}")
    
    # 2. 加权平均集成
    weights = [0.4, 0.35, 0.25]  # 给不同模型不同权重
    weighted_pred = (pred1_vals * weights[0] + 
                     pred2_vals * weights[1] + 
                     pred3_vals * weights[2])
    print(f"加权平均集成前5个预测: {weighted_pred[:5]}")
    print(f"权重分配: 模型1={weights[0]}, 模型2={weights[1]}, 模型3={weights[2]}")
    
    # 3. 计算简单的相关性（如果没有NaN）
    if not (np.isnan(pred1_vals).any() or np.isnan(pred2_vals).any() or np.isnan(pred3_vals).any()):
        corr_12 = np.corrcoef(pred1_vals, pred2_vals)[0, 1]
        corr_13 = np.corrcoef(pred1_vals, pred3_vals)[0, 1]
        corr_23 = np.corrcoef(pred2_vals, pred3_vals)[0, 1]
        
        print(f"\n模型间相关性:")
        print(f"模型1 vs 模型2: {corr_12:.4f}")
        print(f"模型1 vs 模型3: {corr_13:.4f}")
        print(f"模型2 vs 模型3: {corr_23:.4f}")
    
    print("\n=== 集成演示完成 ===")
    print("成功演示了多种集成方法：")
    print("1. 简单平均集成")
    print("2. 加权平均集成")
    print("3. 不同参数的基础模型组合")

if __name__ == "__main__":
    main()