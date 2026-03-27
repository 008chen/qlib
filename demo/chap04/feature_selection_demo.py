#!/usr/bin/env python3
"""
特征选择演示
基于第4章特征工程内容
"""

import qlib
import numpy as np
import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

def filter_feature_selection(X, y, method='f_regression', k=50):
    """过滤法特征选择"""
    # 确保没有NaN值
    assert not X.isna().any().any(), "特征数据包含NaN值"
    assert not y.isna().any(), "标签数据包含NaN值"
    
    if method == 'f_regression':
        # 选择得分最高的k个特征
        selector = SelectKBest(score_func=f_regression, k=k)
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    # 3. 拟合并转换数据
    X_selected = selector.fit_transform(X, y)
    
    # 4. 查看被选中的特征
    selected_features = X.columns[selector.get_support()].tolist()
    
    return X_selected, selected_features, selector.scores_

def wrapper_feature_selection(X, y, n_features=50):
    """包装法特征选择"""
    # 确保没有NaN值
    assert not X.isna().any().any(), "特征数据包含NaN值"
    assert not y.isna().any(), "标签数据包含NaN值"
    
    estimator = LinearRegression()
    # RFE 是一种包装式 (Wrapper) 特征选择方法。它的核心思想是递归地构建模型，并根据模型的表现逐步剔除最不重要的特征，直到达到预设的特征数量
    # n_features_to_select: 最终要保留的特征数量
    selector = RFE(estimator, n_features_to_select=n_features)
    
    
    # 拟合并转换数据
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.support_].tolist()
    
    return X_selected, selected_features

def embedded_feature_selection(X, y, method='lasso', threshold=0.01):
    """嵌入法特征选择"""
    # 确保没有NaN值
    assert not X.isna().any().any(), "特征数据包含NaN值"
    assert not y.isna().any(), "标签数据包含NaN值"
    
    if method == 'lasso':
        model = Lasso(alpha=0.01, max_iter=1000)
    elif method == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    model.fit(X, y)
    
    if method == 'lasso':
        # 线性模型：基于系数绝对值
        feature_importance = np.abs(model.coef_)
    else:
        # 随机森林：基于特征重要性
        feature_importance = model.feature_importances_
    
    # 选择重要性大于阈值的特征
    selected_mask = feature_importance > threshold
    selected_features = X.columns[selected_mask].tolist()
    X_selected = X.iloc[:, selected_mask]
    
    return X_selected, selected_features, feature_importance

def calculate_ic(predictions, returns):
    """计算IC值"""
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_clean = predictions[valid_mask]
    ret_clean = returns[valid_mask]
    
    if len(pred_clean) <= 1:
        return np.nan
    
    correlation = np.corrcoef(pred_clean, ret_clean)[0, 1]
    return correlation

def main():
    print("准备数据...")
    
    # 准备数据处理器（使用较小数据集提高演示速度）
    handler = Alpha158(
        instruments=['SH600000', 'SH600036', 'SH600519'],  # 只使用3只股票
        start_time='2020-01-01',
        end_time='2020-06-30',  # 缩短时间范围
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
    
    # 准备数据
    train_data = dataset.prepare('train')
    test_data = dataset.prepare('test')
    
    # 分离特征和标签（使用正确的DataFrame访问方式）
    label_cols = [col for col in train_data.columns if 'LABEL' in str(col)]
    feature_cols = [col for col in train_data.columns if 'LABEL' not in str(col)]
    
    X_train = train_data[feature_cols]
    y_train = train_data[label_cols[0]]  # 取第一个标签列
    X_test = test_data[feature_cols]
    y_test = test_data[label_cols[0]]
    
    # 处理NaN值 - 更robust的处理方式
    print("处理缺失值...")
    
    # 检查初始NaN情况
    print(f"训练特征NaN数量: {X_train.isna().sum().sum()}")
    print(f"训练标签NaN数量: {y_train.isna().sum()}")
    
    # 处理特征的NaN值
    # 先用均值填充，如果均值是NaN则用0填充
    X_train_filled = X_train.fillna(X_train.mean())
    X_train_filled = X_train_filled.fillna(0)  # 防止均值本身是NaN
    
    # 使用训练集的统计量填充测试集
    X_test_filled = X_test.fillna(X_train.mean())
    X_test_filled = X_test_filled.fillna(0)
    
    # 处理标签的NaN值
    y_train_filled = y_train.fillna(y_train.median())  # 用中位数填充标签
    y_test_filled = y_test.fillna(y_train.median())
    
    # 验证NaN已被清除
    assert not X_train_filled.isna().any().any(), "训练特征仍有NaN值"
    assert not X_test_filled.isna().any().any(), "测试特征仍有NaN值"  
    assert not y_train_filled.isna().any(), "训练标签仍有NaN值"
    assert not y_test_filled.isna().any(), "测试标签仍有NaN值"
    
    # 更新变量
    X_train = X_train_filled
    X_test = X_test_filled
    y_train = y_train_filled
    y_test = y_test_filled
    
    print("缺失值处理完成")
    
    print(f"原始特征数量: {X_train.shape[1]}")
    print(f"训练样本数量: {X_train.shape[0]}")
    
    # 基线模型（使用所有特征）
    print("\n=== 基线模型（所有特征）===")
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_ic = calculate_ic(baseline_pred, y_test.values)
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    
    print(f"基线模型 IC: {baseline_ic:.4f}")
    print(f"基线模型 MSE: {baseline_mse:.6f}")
    
    print("\n=== 过滤法特征选择 ===")
    
    # 过滤法特征选择（减少测试数量以提高速度）
    k_values = [20, 40]
    
    for k in k_values:
        print(f"\n选择前 {k} 个特征:")
        
        X_train_selected, selected_features, scores = filter_feature_selection(
            X_train, y_train, k=k
        )
        X_test_selected = X_test[selected_features]
        
        print(f"选择的特征数量: {len(selected_features)}")
        print(f"前5个特征: {selected_features[:5]}")
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train_selected, y_train)
        pred = model.predict(X_test_selected)
        
        ic = calculate_ic(pred, y_test.values)
        mse = mean_squared_error(y_test, pred)
        
        print(f"过滤法 IC: {ic:.4f}")
        print(f"过滤法 MSE: {mse:.6f}")
    
    print("\n=== 包装法特征选择 ===")
    
    # 包装法特征选择（减少测试数量以提高速度）
    n_features_values = [20]
    
    for n_features in n_features_values:
        print(f"\n使用RFE选择 {n_features} 个特征:")
        
        X_train_selected, selected_features = wrapper_feature_selection(
            X_train, y_train, n_features=n_features
        )
        X_test_selected = X_test[selected_features]
        
        print(f"选择的特征数量: {len(selected_features)}")
        print(f"前5个特征: {selected_features[:5]}")
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train_selected, y_train)
        pred = model.predict(X_test_selected)
        
        ic = calculate_ic(pred, y_test.values)
        mse = mean_squared_error(y_test, pred)
        
        print(f"包装法 IC: {ic:.4f}")
        print(f"包装法 MSE: {mse:.6f}")
    
    print("\n=== 嵌入法特征选择 ===")
    
    # 1. Lasso特征选择
    print("\n1. Lasso特征选择:")
    
    thresholds = [0.01]  # 只测试一个阈值以提高速度
    
    for threshold in thresholds:
        print(f"\n阈值 = {threshold}:")
        
        X_train_selected, selected_features, importance = embedded_feature_selection(
            X_train, y_train, method='lasso', threshold=threshold
        )
        
        if len(selected_features) == 0:
            print("没有特征被选中，阈值可能过高")
            continue
            
        X_test_selected = X_test[selected_features]
        
        print(f"选择的特征数量: {len(selected_features)}")
        print(f"前5个特征: {selected_features[:5]}")
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train_selected, y_train)
        pred = model.predict(X_test_selected)
        
        ic = calculate_ic(pred, y_test.values)
        mse = mean_squared_error(y_test, pred)
        
        print(f"Lasso选择 IC: {ic:.4f}")
        print(f"Lasso选择 MSE: {mse:.6f}")
    
    # 2. 随机森林特征选择
    print("\n2. 随机森林特征选择:")
    
    X_train_selected, selected_features, importance = embedded_feature_selection(
        X_train, y_train, method='random_forest', threshold=0.001
    )
    
    if len(selected_features) > 0:
        X_test_selected = X_test[selected_features]
        
        print(f"选择的特征数量: {len(selected_features)}")
        print(f"前5个重要特征: {selected_features[:5]}")
        
        # 显示特征重要性排序
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("前10个重要特征:")
        print(feature_importance_df.head(10))
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train_selected, y_train)
        pred = model.predict(X_test_selected)
        
        ic = calculate_ic(pred, y_test.values)
        mse = mean_squared_error(y_test, pred)
        
        print(f"随机森林选择 IC: {ic:.4f}")
        print(f"随机森林选择 MSE: {mse:.6f}")
    
    print("\n特征选择演示完成！")

if __name__ == "__main__":
    main()