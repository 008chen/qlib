import qlib
from qlib.data import D
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def calculate_completeness(data):
    """计算数据完整性指标"""
    print("=== 数据完整性评估 ===")
    
    total_elements = data.size
    non_null_elements = data.count().sum()
    completeness = non_null_elements / total_elements
    
    print(f"总数据点: {total_elements:,}")
    print(f"非空数据点: {non_null_elements:,}")
    print(f"数据完整性: {completeness:.4f} ({completeness*100:.2f}%)")
    
    # 按列分析缺失情况
    missing_stats = data.isnull().sum()
    missing_ratio = missing_stats / len(data)
    
    print(f"\n缺失值统计:")
    for col in missing_ratio.index:
        if missing_ratio[col] > 0:
            print(f"  {col}: {missing_stats[col]} ({missing_ratio[col]:.2%})")
    
    # 完整性等级评估
    if completeness >= 0.95:
        grade = "优秀"
    elif completeness >= 0.90:
        grade = "良好"
    elif completeness >= 0.80:
        grade = "一般"
    else:
        grade = "较差"
    
    print(f"\n完整性评级: {grade}")
    
    return {
        'completeness': completeness,
        'missing_stats': missing_stats,
        'missing_ratio': missing_ratio,
        'grade': grade
    }

def calculate_consistency(data):
    """计算数据一致性指标"""
    print("\n=== 数据一致性评估 ===")
    
    consistency_issues = 0
    total_checks = 0
    
    # 1. 价格逻辑一致性检查
    if all(col in data.columns for col in ['$open', '$high', '$low', '$close']):
        print("1. OHLC价格逻辑检查:")
        
        # 高价应该 >= 开盘价和收盘价
        valid_high = (data['$high'] >= data[['$open', '$close']].max(axis=1)).sum()
        invalid_high = len(data) - valid_high
        
        # 低价应该 <= 开盘价和收盘价  
        valid_low = (data['$low'] <= data[['$open', '$close']].min(axis=1)).sum()
        invalid_low = len(data) - valid_low
        
        print(f"   高价逻辑错误: {invalid_high} 条")
        print(f"   低价逻辑错误: {invalid_low} 条")
        
        consistency_issues += invalid_high + invalid_low
        total_checks += 2 * len(data)
    
    # 2. 成交量一致性检查
    if '$volume' in data.columns:
        print("\n2. 成交量一致性检查:")
        negative_volume = (data['$volume'] < 0).sum()
        zero_volume = (data['$volume'] == 0).sum()
        
        print(f"   负成交量: {negative_volume} 条")
        print(f"   零成交量: {zero_volume} 条")
        
        consistency_issues += negative_volume
        total_checks += len(data)
    
    # 3. 数据类型一致性
    print("\n3. 数据类型一致性:")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    print(f"   数值型字段: {len(numeric_cols)}")
    print(f"   总字段数: {len(data.columns)}")
    
    # 计算总体一致性分数
    if total_checks > 0:
        consistency_score = 1 - (consistency_issues / total_checks)
    else:
        consistency_score = 1.0
    
    print(f"\n数据一致性分数: {consistency_score:.4f} ({consistency_score*100:.2f}%)")
    
    return {
        'consistency_score': consistency_score,
        'consistency_issues': consistency_issues,
        'total_checks': total_checks
    }

def calculate_timeliness(data, expected_freq='D'):
    """计算数据时效性指标"""
    print("\n=== 数据时效性评估 ===")
    
    if hasattr(data.index, 'get_level_values'):
        # 多级索引，获取时间索引
        time_index = data.index.get_level_values('datetime')
    else:
        time_index = data.index
    
    # 转换为时间序列
    if not isinstance(time_index, pd.DatetimeIndex):
        print("警告: 索引不是时间类型，跳过时效性检查")
        return {'timeliness_score': 1.0, 'freq_consistency': 1.0}
    
    # 检查数据频率
    time_diff = pd.Series(time_index).diff().dropna()
    
    if expected_freq == 'D':
        expected_diff = pd.Timedelta(days=1)
        tolerance = pd.Timedelta(days=3)  # 考虑周末和节假日
    else:
        expected_diff = pd.Timedelta(expected_freq)
        tolerance = expected_diff * 2
    
    # 计算频率一致性
    freq_matches = (time_diff <= tolerance).sum()
    freq_consistency = freq_matches / len(time_diff) if len(time_diff) > 0 else 1.0
    
    print(f"时间间隔统计:")
    print(f"  期望间隔: {expected_diff}")
    print(f"  实际间隔范围: {time_diff.min()} - {time_diff.max()}")
    print(f"  频率一致性: {freq_consistency:.4f} ({freq_consistency*100:.2f}%)")
    
    # 检查数据新鲜度（假设当前日期为分析基准）
    if len(time_index) > 0:
        latest_date = time_index.max()
        current_date = pd.Timestamp.now().normalize()
        data_lag = (current_date - latest_date).days
        
        print(f"  最新数据日期: {latest_date.strftime('%Y-%m-%d')}")
        print(f"  数据滞后天数: {data_lag}")
        
        # 新鲜度评分
        if data_lag <= 1:
            freshness_score = 1.0
        elif data_lag <= 7:
            freshness_score = 0.8
        elif data_lag <= 30:
            freshness_score = 0.6
        else:
            freshness_score = 0.4
    else:
        freshness_score = 0.0
    
    timeliness_score = (freq_consistency + freshness_score) / 2
    
    print(f"  新鲜度分数: {freshness_score:.4f}")
    print(f"  时效性总分: {timeliness_score:.4f}")
    
    return {
        'timeliness_score': timeliness_score,
        'freq_consistency': freq_consistency,
        'freshness_score': freshness_score,
        'data_lag_days': data_lag if 'data_lag' in locals() else None
    }

def calculate_feature_correlation(features, labels):
    """计算特征与标签相关性"""
    print("\n=== 特征相关性分析 ===")
    
    if labels.ndim > 1:
        labels = labels.iloc[:, 0]  # 使用第一个标签
    
    correlations = {}
    for col in features.columns:
        try:
            corr = features[col].corr(labels)
            if not np.isnan(corr):
                correlations[col] = corr
        except:
            pass
    
    corr_series = pd.Series(correlations)
    
    print(f"特征数量: {len(features.columns)}")
    print(f"有效相关性计算: {len(correlations)}")
    
    # 相关性统计
    print(f"\n相关性统计:")
    print(f"  平均绝对相关性: {corr_series.abs().mean():.4f}")
    print(f"  最大绝对相关性: {corr_series.abs().max():.4f}")
    print(f"  相关性标准差: {corr_series.abs().std():.4f}")
    
    # Top相关特征
    top_corr = corr_series.abs().nlargest(10)
    print(f"\nTop 10 相关特征:")
    for i, (feature, corr) in enumerate(top_corr.items(), 1):
        print(f"  {i:2d}. {feature}: {corr_series[feature]:.4f}")
    
    return corr_series

def calculate_feature_importance(features, labels):
    """计算特征重要性"""
    print("\n=== 特征重要性分析 ===")
    
    # 处理缺失值
    features_clean = features.fillna(features.mean())
    
    if labels.ndim > 1:
        labels_clean = labels.iloc[:, 0].fillna(labels.iloc[:, 0].mean())
    else:
        labels_clean = labels.fillna(labels.mean())
    
    # 确保数据对齐
    common_index = features_clean.index.intersection(labels_clean.index)
    features_clean = features_clean.loc[common_index]
    labels_clean = labels_clean.loc[common_index]
    
    try:
        # 随机森林特征重要性
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(features_clean, labels_clean)
        
        rf_importance = pd.Series(rf.feature_importances_, index=features_clean.columns)
        rf_importance = rf_importance.sort_values(ascending=False)
        
        print(f"随机森林特征重要性 (Top 10):")
        for i, (feature, importance) in enumerate(rf_importance.head(10).items(), 1):
            print(f"  {i:2d}. {feature}: {importance:.4f}")
        
    except Exception as e:
        print(f"随机森林特征重要性计算失败: {e}")
        rf_importance = pd.Series()
    
    try:
        # 互信息特征重要性
        mi_scores = mutual_info_regression(features_clean, labels_clean, random_state=42)
        mi_importance = pd.Series(mi_scores, index=features_clean.columns)
        mi_importance = mi_importance.sort_values(ascending=False)
        
        print(f"\n互信息特征重要性 (Top 10):")
        for i, (feature, importance) in enumerate(mi_importance.head(10).items(), 1):
            print(f"  {i:2d}. {feature}: {importance:.4f}")
            
    except Exception as e:
        print(f"互信息特征重要性计算失败: {e}")
        mi_importance = pd.Series()
    
    return {
        'rf_importance': rf_importance,
        'mi_importance': mi_importance
    }

def calculate_feature_stability(features, time_periods=None):
    """计算特征稳定性"""
    print("\n=== 特征稳定性分析 ===")
    
    if time_periods is None:
        # 自动分割时间段
        if hasattr(features.index, 'get_level_values'):
            dates = features.index.get_level_values('datetime').unique()
        else:
            dates = features.index.unique()
        
        dates = sorted(dates)
        n_periods = min(4, len(dates) // 20)  # 至少20个观测值一个时期
        
        if n_periods < 2:
            print("数据量不足，无法进行稳定性分析")
            return {}
        
        time_periods = [dates[i * len(dates) // n_periods:(i + 1) * len(dates) // n_periods] 
                       for i in range(n_periods)]
    
    stability_scores = {}
    
    numeric_features = features.select_dtypes(include=[np.number]).columns[:10]  # 限制特征数量
    print(f"分析特征数量: {len(numeric_features)}")
    
    for col in numeric_features:
        feature_data = features[col].dropna()
        
        if len(feature_data) < 20:  # 数据量太少
            continue
        
        period_stats = []
        for period in time_periods:
            if hasattr(features.index, 'get_level_values'):
                mask = features.index.get_level_values('datetime').isin(period)
            else:
                mask = features.index.isin(period)
            
            period_data = feature_data[mask]
            if len(period_data) > 5:
                period_stats.append({
                    'mean': period_data.mean(),
                    'std': period_data.std(),
                    'skew': period_data.skew()
                })
        
        if len(period_stats) >= 2:
            # 计算不同时期的统计量稳定性
            means = [s['mean'] for s in period_stats]
            stds = [s['std'] for s in period_stats if not np.isnan(s['std'])]
            
            mean_stability = 1 - (np.std(means) / np.mean(np.abs(means)) if np.mean(np.abs(means)) > 0 else 0)
            std_stability = 1 - (np.std(stds) / np.mean(stds) if len(stds) > 0 and np.mean(stds) > 0 else 0)
            
            stability_scores[col] = (mean_stability + std_stability) / 2
    
    stability_series = pd.Series(stability_scores).sort_values(ascending=False)
    
    print(f"特征稳定性分数 (Top 10):")
    for i, (feature, score) in enumerate(stability_series.head(10).items(), 1):
        print(f"  {i:2d}. {feature}: {score:.4f}")
    
    return stability_series

def detect_outliers(data, method='iqr'):
    """异常值检测"""
    print(f"\n=== 异常值检测 ({method.upper()}方法) ===")
    
    outliers = {}
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        col_data = data[col].dropna()
        
        if len(col_data) < 10:
            continue
            
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outlier_mask = z_scores > 3
            
        outliers[col] = outlier_mask.sum()
    
    outlier_series = pd.Series(outliers)
    total_outliers = outlier_series.sum()
    
    print(f"异常值统计:")
    print(f"  总异常值数量: {total_outliers}")
    print(f"  异常值比例: {total_outliers / len(data):.4f} ({total_outliers / len(data) * 100:.2f}%)")
    
    # 异常值最多的特征
    top_outliers = outlier_series.nlargest(10)
    if len(top_outliers) > 0:
        print(f"\n异常值最多的特征:")
        for feature, count in top_outliers.items():
            if count > 0:
                print(f"  {feature}: {count} ({count / len(data) * 100:.2f}%)")
    
    return outlier_series

def generate_quality_report(data, labels=None):
    """生成综合数据质量报告"""
    print("\n" + "="*50)
    print("        数据质量综合评估报告")
    print("="*50)
    
    # 基本信息
    print(f"\n数据基本信息:")
    print(f"  数据形状: {data.shape}")
    print(f"  数据类型: {len(data.dtypes.unique())} 种")
    print(f"  内存使用: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # 1. 完整性评估
    completeness_result = calculate_completeness(data)
    
    # 2. 一致性评估
    consistency_result = calculate_consistency(data)
    
    # 3. 时效性评估
    timeliness_result = calculate_timeliness(data)
    
    # 4. 异常值检测
    outliers_iqr = detect_outliers(data, 'iqr')
    outliers_zscore = detect_outliers(data, 'zscore')
    
    # 5. 特征质量分析（如果有标签）  
    if labels is not None:
        feature_corr = calculate_feature_correlation(data, labels)
        feature_importance = calculate_feature_importance(data, labels)
        feature_stability = calculate_feature_stability(data)
    
    # 综合评分
    scores = {
        'completeness': completeness_result['completeness'],
        'consistency': consistency_result['consistency_score'],
        'timeliness': timeliness_result['timeliness_score'],
    }
    
    overall_score = np.mean(list(scores.values()))
    
    print(f"\n" + "="*50)
    print("        综合质量评分")
    print("="*50)
    
    for metric, score in scores.items():
        print(f"  {metric.capitalize():15s}: {score:.3f} ({score*100:.1f}%)")
    
    print(f"  {'Overall Score':15s}: {overall_score:.3f} ({overall_score*100:.1f}%)")
    
    # 评级
    if overall_score >= 0.9:
        grade = "A (优秀)"
    elif overall_score >= 0.8:
        grade = "B (良好)"
    elif overall_score >= 0.7:
        grade = "C (一般)"
    elif overall_score >= 0.6:
        grade = "D (较差)"
    else:
        grade = "F (不合格)"
    
    print(f"  {'Data Quality Grade':15s}: {grade}")
    
    return {
        'overall_score': overall_score,
        'scores': scores,
        'grade': grade,
        'completeness': completeness_result,
        'consistency': consistency_result,
        'timeliness': timeliness_result,
        'outliers': {'iqr': outliers_iqr, 'zscore': outliers_zscore}
    }

if __name__ == "__main__":
    try:
        # 初始化Qlib
        qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")
        
        # 获取测试数据
        print("正在获取测试数据...")
        data = D.features(
            instruments=['SH600000', 'SH600036'],
            fields=['$close', '$open', '$high', '$low', '$volume', 
                   'Mean($close, 20)', 'Std($close, 20)', '$close / Ref($close, 1) - 1'],
            start_time='2020-01-01',
            end_time='2020-03-31',
            freq='day'
        )
        
        # 生成标签（未来收益率）
        if hasattr(data.index, 'get_level_values'):
            labels = D.features(
                instruments=['SH600000', 'SH600036'],
                fields=['Ref($close, -1) / $close - 1'],
                start_time='2020-01-01',
                end_time='2020-03-31',
                freq='day'
            )
        else:
            # 简化版标签
            labels = data[['$close']].shift(-1) / data[['$close']] - 1
            labels.columns = ['future_return']
        
        # 生成综合质量报告
        report = generate_quality_report(data, labels)
        
        print(f"\n" + "="*50)
        print("        数据质量评估完成")
        print("="*50)
        print("\n建议:")
        
        if report['overall_score'] >= 0.9:
            print("✓ 数据质量优秀，可直接用于建模")
        elif report['overall_score'] >= 0.8:
            print("✓ 数据质量良好，建议少量清洗后使用")
        elif report['overall_score'] >= 0.7:
            print("⚠ 数据质量一般，需要进一步清洗和预处理")
        else:
            print("✗ 数据质量较差，需要大量数据清洗工作")
        
        print("\n具体改进建议:")
        if report['completeness']['completeness'] < 0.9:
            print("- 处理缺失值问题")
        if report['consistency']['consistency_score'] < 0.9:
            print("- 修复数据一致性问题") 
        if report['timeliness']['timeliness_score'] < 0.9:
            print("- 检查数据时效性")
            
    except Exception as e:
        print(f"数据质量评估失败: {e}")
        print("请检查:")
        print("1. Qlib是否正确安装和配置")
        print("2. 数据是否可用")
        print("3. 网络连接是否正常")