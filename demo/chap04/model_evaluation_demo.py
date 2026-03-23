#!/usr/bin/env python3
"""
模型评估指标演示
基于第4章模型评估指标内容
"""

import qlib
import numpy as np
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

def calculate_ic(predictions, returns):
    """计算IC值"""
    # 移除缺失值
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_clean = predictions[valid_mask]
    ret_clean = returns[valid_mask]
    
    if len(pred_clean) <= 1:
        return np.nan
    
    # 计算相关系数
    ic, p_value = pearsonr(pred_clean, ret_clean)
    
    return ic

def calculate_rank_ic(predictions, returns):
    """计算Rank IC"""
    # 移除缺失值
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_clean = predictions[valid_mask]
    ret_clean = returns[valid_mask]
    
    if len(pred_clean) <= 1:
        return np.nan
    
    # 计算Spearman相关系数
    rank_ic, p_value = spearmanr(pred_clean, ret_clean)
    
    return rank_ic

def calculate_time_series_ic(predictions, returns, dates):
    """计算时间序列IC"""
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'prediction': predictions,
        'return': returns
    })
    
    # 按日期分组计算IC
    ic_series = df.groupby('date').apply(
        lambda x: calculate_ic(x['prediction'].values, x['return'].values)
    ).dropna()
    
    return ic_series

def calculate_time_series_rank_ic(predictions, returns, dates):
    """计算时间序列Rank IC"""
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'prediction': predictions,
        'return': returns
    })
    
    # 按日期分组计算Rank IC
    rank_ic_series = df.groupby('date').apply(
        lambda x: calculate_rank_ic(x['prediction'].values, x['return'].values)
    ).dropna()
    
    return rank_ic_series

def calculate_icir(ic_series):
    """计算ICIR"""
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    
    if std_ic == 0 or np.isnan(std_ic):
        return np.nan
    
    icir = mean_ic / std_ic
    return icir

def comprehensive_model_evaluation(predictions, returns, dates):
    """综合模型评估"""
    # 计算各种IC指标
    ic_series = calculate_time_series_ic(predictions, returns, dates)
    rank_ic_series = calculate_time_series_rank_ic(predictions, returns, dates)
    
    # 计算统计指标
    evaluation_metrics = {
        'Mean_IC': ic_series.mean(),
        'Std_IC': ic_series.std(),
        'ICIR': calculate_icir(ic_series),
        'Mean_Rank_IC': rank_ic_series.mean(),
        'Std_Rank_IC': rank_ic_series.std(),
        'Rank_ICIR': calculate_icir(rank_ic_series),
        'IC_Positive_Rate': (ic_series > 0).mean(),
        'Rank_IC_Positive_Rate': (rank_ic_series > 0).mean(),
        'Max_IC': ic_series.max(),
        'Min_IC': ic_series.min(),
        'Max_Rank_IC': rank_ic_series.max(),
        'Min_Rank_IC': rank_ic_series.min()
    }
    
    return evaluation_metrics, ic_series, rank_ic_series

def plot_ic_series(ic_series, rank_ic_series, title="IC时间序列"):
    """绘制IC时间序列"""
    try:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(ic_series.index, ic_series.values, label='IC', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=ic_series.mean(), color='g', linestyle='-', alpha=0.5, label=f'Mean IC: {ic_series.mean():.4f}')
        plt.title('IC时间序列')
        plt.ylabel('IC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(rank_ic_series.index, rank_ic_series.values, label='Rank IC', alpha=0.7, color='orange')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=rank_ic_series.mean(), color='g', linestyle='-', alpha=0.5, label=f'Mean Rank IC: {rank_ic_series.mean():.4f}')
        plt.title('Rank IC时间序列')
        plt.xlabel('日期')
        plt.ylabel('Rank IC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ic_time_series.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("IC时间序列图已保存为 ic_time_series.png")
    except Exception as e:
        print(f"绘图功能在当前环境中不可用: {e}")

def main():
    print("准备数据...")
    
    # 准备数据处理器
    handler = Alpha158(
        instruments='csi300',
        start_time='2020-01-01',
        end_time='2020-12-31',
        fit_start_time='2020-01-01',
        fit_end_time='2020-08-31',
        freq='day'
    )
    
    # 使用DatasetH管理数据分割
    from qlib.data.dataset import DatasetH
    dataset = DatasetH(
        handler=handler,
        segments={
            'train': ('2020-01-01', '2020-08-31'),
            'test': ('2020-09-01', '2020-12-31')
        }
    )
    
    # 获取数据用于展示
    train_data = dataset.prepare('train')
    test_data = dataset.prepare('test')
    
    # 分离特征和标签用于某些算法
    X_train = train_data.drop('LABEL0', axis=1)
    y_train = train_data['LABEL0']
    X_test = test_data.drop('LABEL0', axis=1)
    y_test = test_data['LABEL0']
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    print("\n=== 训练多个模型进行对比 ===")
    
    # 模型1: LightGBM
    print("训练LightGBM模型...")
    lgb_model = LGBModel(
        loss='mse',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        verbose=-1
    )
    lgb_model.fit(dataset)
    lgb_pred = lgb_model.predict(dataset, segment='test')
    
    # 模型2: LightGBM with different params
    print("训练LightGBM模型2...")
    lgb_model2 = LGBModel(
        loss='mse',
        n_estimators=50,
        max_depth=8,
        learning_rate=0.2,
        verbose=-1
    )
    lgb_model2.fit(dataset)
    lgb_pred2 = lgb_model2.predict(dataset, segment='test')
    
    # 准备评估数据
    test_dates = X_test.index.get_level_values('datetime')
    
    print("\n=== 单个模型评估 ===")
    
    # 评估模型1 - 将预测结果转换为数组格式
    print("模型1评估:")
    if hasattr(lgb_pred, 'values'):
        pred1_values = lgb_pred.values.flatten() if hasattr(lgb_pred.values, 'flatten') else lgb_pred.values
    else:
        pred1_values = lgb_pred.flatten() if hasattr(lgb_pred, 'flatten') else lgb_pred
    
    metrics1, ic_series1, rank_ic_series1 = comprehensive_model_evaluation(
        pred1_values, y_test.values, test_dates
    )
    
    for metric, value in metrics1.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n模型2评估:")
    # 将预测结果转换为数组格式
    if hasattr(lgb_pred2, 'values'):
        pred2_values = lgb_pred2.values.flatten() if hasattr(lgb_pred2.values, 'flatten') else lgb_pred2.values
    else:
        pred2_values = lgb_pred2.flatten() if hasattr(lgb_pred2, 'flatten') else lgb_pred2
    
    metrics2, ic_series2, rank_ic_series2 = comprehensive_model_evaluation(
        pred2_values, y_test.values, test_dates
    )
    
    for metric, value in metrics2.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== 模型对比分析 ===")
    
    # 创建对比表格
    comparison_df = pd.DataFrame({
        '模型1': metrics1,
        '模型2': metrics2
    })
    
    print("模型性能对比:")
    print(comparison_df)
    
    # 详细分析
    print("\n=== 详细分析 ===")
    
    print("1. IC分析:")
    print(f"模型1 IC分布: 均值={ic_series1.mean():.4f}, 标准差={ic_series1.std():.4f}")
    print(f"模型2 IC分布: 均值={ic_series2.mean():.4f}, 标准差={ic_series2.std():.4f}")
    
    print("\n2. IC稳定性分析:")
    ic1_positive_days = (ic_series1 > 0).sum()
    ic2_positive_days = (ic_series2 > 0).sum()
    total_days = len(ic_series1)
    
    print(f"模型1 IC正值天数: {ic1_positive_days}/{total_days} ({ic1_positive_days/total_days*100:.1f}%)")
    print(f"模型2 IC正值天数: {ic2_positive_days}/{total_days} ({ic2_positive_days/total_days*100:.1f}%)")
    
    print("\n3. IC分位数分析:")
    ic1_quantiles = ic_series1.quantile([0.25, 0.5, 0.75])
    ic2_quantiles = ic_series2.quantile([0.25, 0.5, 0.75])
    
    print("模型1 IC分位数:")
    print(f"25%: {ic1_quantiles[0.25]:.4f}, 50%: {ic1_quantiles[0.5]:.4f}, 75%: {ic1_quantiles[0.75]:.4f}")
    
    print("模型2 IC分位数:")
    print(f"25%: {ic2_quantiles[0.25]:.4f}, 50%: {ic2_quantiles[0.5]:.4f}, 75%: {ic2_quantiles[0.75]:.4f}")
    
    print("\n4. 时间段分析:")
    # 分月份分析
    ic1_monthly = ic_series1.groupby(ic_series1.index.month).mean()
    ic2_monthly = ic_series2.groupby(ic_series2.index.month).mean()
    
    print("月度IC均值对比:")
    monthly_comparison = pd.DataFrame({
        '模型1': ic1_monthly,
        '模型2': ic2_monthly
    })
    print(monthly_comparison)
    
    print("\n=== IC相关性分析 ===")
    
    # 两个模型IC的相关性
    ic_correlation = np.corrcoef(ic_series1, ic_series2)[0, 1]
    print(f"两个模型IC的相关系数: {ic_correlation:.4f}")
    
    # 预测值相关性
    pred_correlation = np.corrcoef(lgb_pred, lgb_pred2)[0, 1]
    print(f"两个模型预测值的相关系数: {pred_correlation:.4f}")
    
    print("\n=== 风险调整后收益分析 ===")
    
    # 计算夏普比率相似的指标（ICIR）
    print(f"模型1 ICIR: {metrics1['ICIR']:.4f}")
    print(f"模型2 ICIR: {metrics2['ICIR']:.4f}")
    
    # 最大回撤（基于IC）
    def calculate_ic_drawdown(ic_series):
        cumulative_ic = ic_series.cumsum()
        rolling_max = cumulative_ic.expanding().max()
        drawdown = cumulative_ic - rolling_max
        max_drawdown = drawdown.min()
        return max_drawdown
    
    ic1_max_drawdown = calculate_ic_drawdown(ic_series1)
    ic2_max_drawdown = calculate_ic_drawdown(ic_series2)
    
    print(f"模型1 IC最大回撤: {ic1_max_drawdown:.4f}")
    print(f"模型2 IC最大回撤: {ic2_max_drawdown:.4f}")
    
    # 绘制IC时间序列图
    print("\n=== 绘制IC时间序列图 ===")
    plot_ic_series(ic_series1, rank_ic_series1, "模型1 IC时间序列")
    
    print("\n=== 评估总结 ===")
    
    # 综合评分
    def calculate_composite_score(metrics):
        score = (
            metrics['Mean_IC'] * 0.3 +
            metrics['ICIR'] * 0.3 +
            metrics['IC_Positive_Rate'] * 0.2 +
            (1 - abs(metrics['Min_IC']) / max(abs(metrics['Max_IC']), abs(metrics['Min_IC']))) * 0.2
        )
        return score
    
    score1 = calculate_composite_score(metrics1)
    score2 = calculate_composite_score(metrics2)
    
    print(f"模型1综合评分: {score1:.4f}")
    print(f"模型2综合评分: {score2:.4f}")
    
    if score1 > score2:
        print("推荐使用模型1")
    else:
        print("推荐使用模型2")
    
    print("\n模型评估演示完成！")

if __name__ == "__main__":
    main()