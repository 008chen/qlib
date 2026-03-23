#!/usr/bin/env python3
"""
超参数优化演示
基于第4章超参数调优内容
"""

import qlib
import numpy as np
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

def calculate_ic(y_true, y_pred):
    """计算IC值用作评分函数"""
    valid_mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    pred_clean = y_pred[valid_mask]
    true_clean = y_true[valid_mask]
    
    if len(pred_clean) <= 1:
        return 0
    
    correlation = np.corrcoef(pred_clean, true_clean)[0, 1]
    return correlation if not np.isnan(correlation) else 0

def grid_search_optimization(X_train, y_train):
    """网格搜索超参数优化"""
    print("执行网格搜索...")
    
    # 定义参数网格（简化版本）
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # 创建模型
    model = RandomForestRegressor(random_state=42)
    
    # 创建评分函数
    ic_scorer = make_scorer(calculate_ic, greater_is_better=True)
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,  # 减少CV折数以提高速度
        scoring=ic_scorer,
        n_jobs=1,  # 避免多进程问题
        verbose=1
    )
    
    # 训练
    grid_search.fit(X_train, y_train)
    
    # 最佳参数
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"网格搜索最佳参数: {best_params}")
    print(f"网格搜索最佳得分: {best_score:.4f}")
    
    return grid_search.best_estimator_, best_params

def random_search_optimization(X_train, y_train):
    """随机搜索超参数优化"""
    print("执行随机搜索...")
    
    # 定义参数分布
    param_distributions = {
        'n_estimators': randint(50, 150),
        'max_depth': randint(5, 15),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None]
    }
    
    # 创建模型
    model = RandomForestRegressor(random_state=42)
    
    # 创建评分函数
    ic_scorer = make_scorer(calculate_ic, greater_is_better=True)
    
    # 随机搜索
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,  # 减少迭代次数
        cv=3,
        scoring=ic_scorer,
        n_jobs=1,
        random_state=42,
        verbose=1
    )
    
    # 训练
    random_search.fit(X_train, y_train)
    
    # 最佳参数
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"随机搜索最佳参数: {best_params}")
    print(f"随机搜索最佳得分: {best_score:.4f}")
    
    return random_search.best_estimator_, best_params

def manual_optimization(dataset, y_val):
    """手动参数优化"""
    print("执行手动参数优化...")
    
    best_ic = -1
    best_params = None
    best_model = None
    
    # 定义参数组合（简化以提高速度）
    param_combinations = [
        {'n_estimators': 50, 'max_depth': 6, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
    ]
    
    for i, params in enumerate(param_combinations):
        print(f"测试参数组合 {i+1}: {params}")
        
        try:
            # 创建模型
            model = LGBModel(
                loss='mse',
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                verbose=-1
            )
            
            # 训练模型
            model.fit(dataset)
            
            # 验证预测
            val_pred = model.predict(dataset, segment='valid')
            if hasattr(val_pred, 'values'):
                val_pred_values = val_pred.values.flatten() if hasattr(val_pred.values, 'flatten') else val_pred.values
            else:
                val_pred_values = val_pred.flatten() if hasattr(val_pred, 'flatten') else val_pred
            val_ic = calculate_ic(y_val.values, val_pred_values)
            
            print(f"验证集IC: {val_ic:.4f}")
            
            if val_ic > best_ic:
                best_ic = val_ic
                best_params = params
                best_model = model
                
        except Exception as e:
            print(f"参数组合 {i+1} 训练失败: {e}")
            continue
    
    print(f"手动优化最佳参数: {best_params}")
    print(f"手动优化最佳IC: {best_ic:.4f}")
    
    return best_model, best_params

def main():
    print("准备数据...")
    
    # 准备数据处理器（使用较小数据集提高演示速度）
    handler = Alpha158(
        instruments=['SH600000', 'SH600036', 'SH600519'],  # 只使用3只股票
        start_time='2020-01-01',
        end_time='2020-06-30',  # 缩短时间范围
        fit_start_time='2020-01-01',
        fit_end_time='2020-04-30',
        freq='day'
    )
    
    # 使用DatasetH管理数据分割
    from qlib.data.dataset import DatasetH
    dataset = DatasetH(
        handler=handler,
        segments={
            'train': ('2020-01-01', '2020-04-30'),
            'valid': ('2020-05-01', '2020-05-31'),
            'test': ('2020-06-01', '2020-06-30')
        }
    )
    
    # 获取数据
    train_data = dataset.prepare('train')
    valid_data = dataset.prepare('valid')
    test_data = dataset.prepare('test')
    
    # 分离特征和标签（使用正确的DataFrame访问方式）
    label_cols = [col for col in train_data.columns if 'LABEL' in str(col)]
    feature_cols = [col for col in train_data.columns if 'LABEL' not in str(col)]
    
    X_train = train_data[feature_cols]
    y_train = train_data[label_cols[0]]
    X_valid = valid_data[feature_cols]
    y_valid = valid_data[label_cols[0]]
    X_test = test_data[feature_cols]
    y_test = test_data[label_cols[0]]
    
    # 数据验证和清理
    print("数据验证和清理...")
    print(f"原始数据大小:")
    print(f"- 训练集: {X_train.shape}")
    print(f"- 验证集: {X_valid.shape}")
    print(f"- 测试集: {X_test.shape}")
    
    # 检查空数据
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("训练集或测试集为空，请检查数据分割设置")
    
    # 处理NaN值
    print("处理缺失值...")
    
    # 训练集NaN处理
    X_train_clean = X_train.fillna(X_train.mean()).fillna(0)
    y_train_clean = y_train.fillna(y_train.median())
    
    # 验证集NaN处理
    X_valid_clean = X_valid.fillna(X_train.mean()).fillna(0)
    y_valid_clean = y_valid.fillna(y_train.median())
    
    # 测试集NaN处理
    X_test_clean = X_test.fillna(X_train.mean()).fillna(0)
    y_test_clean = y_test.fillna(y_train.median())
    
    # 验证清理后的数据
    assert not X_train_clean.isna().any().any(), "训练特征仍有NaN"
    assert not y_train_clean.isna().any(), "训练标签仍有NaN"
    assert not X_test_clean.isna().any().any(), "测试特征仍有NaN"
    assert not y_test_clean.isna().any(), "测试标签仍有NaN"
    
    # 更新变量
    X_train, y_train = X_train_clean, y_train_clean
    X_valid, y_valid = X_valid_clean, y_valid_clean
    X_test, y_test = X_test_clean, y_test_clean
    
    print("数据清理完成")
    print(f"最终数据大小:")
    print(f"- 训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"- 验证集: {X_valid.shape}, 标签: {y_valid.shape}")
    print(f"- 测试集: {X_test.shape}, 标签: {y_test.shape}")
    
    # 基线模型
    print("\n=== 基线模型 ===")
    baseline_model = LGBModel(
        loss='mse',
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        verbose=-1
    )
    
    baseline_model.fit(dataset)
    baseline_pred = baseline_model.predict(dataset, segment='test')
    if hasattr(baseline_pred, 'values'):
        baseline_pred_values = baseline_pred.values.flatten() if hasattr(baseline_pred.values, 'flatten') else baseline_pred.values
    else:
        baseline_pred_values = baseline_pred.flatten() if hasattr(baseline_pred, 'flatten') else baseline_pred
    baseline_ic = calculate_ic(y_test.values, baseline_pred_values)
    baseline_mse = mean_squared_error(y_test, baseline_pred_values)
    
    print(f"基线模型 IC: {baseline_ic:.4f}")
    print(f"基线模型 MSE: {baseline_mse:.6f}")
    
    print("\n=== 网格搜索优化 ===")
    try:
        grid_model, grid_params = grid_search_optimization(X_train, y_train)
        grid_pred = grid_model.predict(X_test)
        grid_ic = calculate_ic(y_test.values, grid_pred)
        grid_mse = mean_squared_error(y_test, grid_pred)
        
        print(f"网格搜索优化后 IC: {grid_ic:.4f}")
        print(f"网格搜索优化后 MSE: {grid_mse:.6f}")
    except Exception as e:
        print(f"网格搜索出现错误: {e}")
    
    print("\n=== 随机搜索优化 ===")
    try:
        random_model, random_params = random_search_optimization(X_train, y_train)
        random_pred = random_model.predict(X_test)
        random_ic = calculate_ic(y_test.values, random_pred)
        random_mse = mean_squared_error(y_test, random_pred)
        
        print(f"随机搜索优化后 IC: {random_ic:.4f}")
        print(f"随机搜索优化后 MSE: {random_mse:.6f}")
    except Exception as e:
        print(f"随机搜索出现错误: {e}")
    
    print("\n=== 手动参数优化 ===")
    manual_model, manual_params = manual_optimization(dataset, y_valid)
    manual_pred = manual_model.predict(dataset, segment='test')
    if hasattr(manual_pred, 'values'):
        manual_pred_values = manual_pred.values.flatten() if hasattr(manual_pred.values, 'flatten') else manual_pred.values
    else:
        manual_pred_values = manual_pred.flatten() if hasattr(manual_pred, 'flatten') else manual_pred
    manual_ic = calculate_ic(y_test.values, manual_pred_values)
    manual_mse = mean_squared_error(y_test, manual_pred_values)
    
    print(f"手动优化后 IC: {manual_ic:.4f}")
    print(f"手动优化后 MSE: {manual_mse:.6f}")
    
    print("\n=== 参数敏感性分析 ===")
    
    # 学习率敏感性
    print("学习率敏感性分析:")
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    
    for lr in learning_rates:
        model = LGBModel(
            loss='mse',
            n_estimators=100,
            max_depth=6,
            learning_rate=lr,
            verbose=-1
        )
        
        model.fit(dataset)
        pred = model.predict(dataset, segment='valid')
        if hasattr(pred, 'values'):
            pred_values = pred.values.flatten() if hasattr(pred.values, 'flatten') else pred.values
        else:
            pred_values = pred.flatten() if hasattr(pred, 'flatten') else pred
        ic = calculate_ic(y_valid.values, pred_values)
        
        print(f"学习率 {lr}: IC = {ic:.4f}")
    
    # 树的数量敏感性
    print("\n树的数量敏感性分析:")
    n_estimators_list = [50, 100, 150, 200]
    
    for n_est in n_estimators_list:
        model = LGBModel(
            loss='mse',
            n_estimators=n_est,
            max_depth=6,
            learning_rate=0.1,
            verbose=-1
        )
        
        model.fit(dataset)
        pred = model.predict(dataset, segment='valid')
        if hasattr(pred, 'values'):
            pred_values = pred.values.flatten() if hasattr(pred.values, 'flatten') else pred.values
        else:
            pred_values = pred.flatten() if hasattr(pred, 'flatten') else pred
        ic = calculate_ic(y_valid.values, pred_values)
        
        print(f"树的数量 {n_est}: IC = {ic:.4f}")
    
    print("\n=== 结果总结 ===")
    
    results = [
        ("基线模型", baseline_ic, baseline_mse),
        ("手动优化", manual_ic, manual_mse)
    ]
    
    print("模型性能对比:")
    print("方法\t\tIC\tMSE")
    print("-" * 30)
    for method, ic, mse in results:
        print(f"{method}\t{ic:.4f}\t{mse:.6f}")
    
    print("\n超参数优化演示完成！")

if __name__ == "__main__":
    main()