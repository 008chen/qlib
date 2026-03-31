import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
import qlib

def find_best_threshold_ic(feature_name, label_name, start_date, end_date, 
                           instruments='csi300', n_thresholds=100):
    """
    搜索特征的最佳阈值，使得特征>阈值与二元label的IC最大
    
    Parameters:
    -----------
    feature_name : str
        特征字段名，如 'CLOSE/OPEN' 或自定义特征
    label_name : str
        label字段名，label取值应为0或1
    start_date, end_date : str
        数据日期范围
    instruments : str
        股票池
    n_thresholds : int
        阈值搜索的粒度
    
    Returns:
    --------
    best_threshold : float
        最优阈值
    best_ic : float
        最优IC值
    ic_series : pd.Series
        不同阈值对应的IC
    """
    
    # 1. 加载数据
    fields = [feature_name, label_name]
    df = D.features(
        instruments=instruments,
        fields=fields,
        start_time=start_date,
        end_time=end_date
    )
    
    print(df)
    # 2. 数据清洗
    df = df.dropna()
    feature = df[feature_name]
    label = df[label_name]
    
    # 确保label是二元0/1
    assert set(label.unique()).issubset({0, 1}), "Label必须是0或1"
    
    # 3. 确定阈值搜索范围
    # 使用特征的分位数作为阈值候选，避免极端值影响
    thresholds = np.linspace(
        feature.quantile(0.05), 
        feature.quantile(0.95), 
        n_thresholds
    )
    
    # 4. 搜索最优阈值
    ic_results = []
    
    for thresh in thresholds:
        # 二值化特征
        binary_feature = (feature > thresh).astype(int)
        
        # 计算IC (Spearman)
        if len(binary_feature.unique()) > 1:  # 避免常数情况
            ic, pvalue = spearmanr(binary_feature, label)
            ic_results.append({
                'threshold': thresh,
                'ic': ic,
                'pvalue': pvalue,
                'positive_rate': binary_feature.mean()  # 特征为1的比例
            })
    
    ic_df = pd.DataFrame(ic_results)
    print(ic_df)
    
    # 5. 找到最优阈值（按绝对IC值最大，或正IC最大）
    best_idx = ic_df['ic'].abs().idxmax()
    best_threshold = ic_df.loc[best_idx, 'threshold']
    best_ic = ic_df.loc[best_idx, 'ic']
    
    return best_threshold, best_ic, ic_df.set_index('threshold')['ic']


def calculate_group_ic(feature_name, label_name, threshold, start_date, end_date,
                      instruments='csi300', groupby='datetime'):
    """
    计算分组的IC（按日期分组计算IC后取平均）
    
    适用于时间序列数据，避免未来信息泄露
    """
    fields = [feature_name, label_name]
    df = D.features(
        instruments=instruments,
        fields=fields,
        start_time=start_date,
        end_time=end_date
    )
    
    df = df.dropna()
    df['binary_feature'] = (df[feature_name] > threshold).astype(int)
    
    # 按日期分组计算IC
    def calc_ic(group):
        if len(group) < 5 or group['binary_feature'].nunique() == 1:
            return np.nan
        ic, _ = spearmanr(group['binary_feature'], group[label_name])
        return ic
    
    ic_by_date = df.groupby(groupby).apply(calc_ic)
    ic_by_date = ic_by_date.dropna()
    
    return {
        'ic_mean': ic_by_date.mean(),
        'ic_std': ic_by_date.std(),
        'ic_ir': ic_by_date.mean() / ic_by_date.std() if ic_by_date.std() != 0 else np.nan,
        'ic_series': ic_by_date,
        'positive_ratio': (ic_by_date > 0).mean()
    }


# ============ 使用示例 ============

if __name__ == "__main__":
   
    
    # 初始化Qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')
    
    
    factor_expr = "100*($close - $low)/($high-$low  + 1e-12)"
    # 定义未来收益 (例如：未来5日收益)
    future_ret_expr = "(Ref($high, -1)-$close) >($close - $low + 1e-12)"
    
    start_time = "2020-01-01"
    end_time = "2026-01-10"
    
    
    instruments = D.instruments('all') 
    # 示例1：搜索最优阈值
    best_thresh, best_ic, ic_series = find_best_threshold_ic(
        feature_name=factor_expr,  # 日内收益率
        label_name=future_ret_expr,          # 你的二元label
        start_date=start_time,
        end_date=end_time,
        instruments=instruments,
        n_thresholds=50
    )
    
    print(f"最优阈值: {best_thresh:.4f}")
    print(f"最优IC: {best_ic:.4f}")
    
    # 示例2：计算该阈值下的分组IC（更严谨）
    group_ic = calculate_group_ic(
        feature_name=factor_expr,  # 日内收益率
        label_name=future_ret_expr,          # 你的二元label
        threshold=best_thresh,
        instruments=instruments。,
        start_date=start_time,
        end_date=end_time,
    )
    
    print(f"\n分组IC均值: {group_ic['ic_mean']:.4f}")
    print(f"分组IC标准差: {group_ic['ic_std']:.4f}")
    print(f"IC_IR: {group_ic['ic_ir']:.4f}")
    print(f"IC为正的比例: {group_ic['positive_ratio']:.2%}")