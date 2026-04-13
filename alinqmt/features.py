import talib
import pandas as pd
import numpy as np




def calc_talib_features(group):
    '''
    单df
        
    '''
    group = group.dropna()
    close = group['close'].values.astype(np.float64)
    high = group['high'].values.astype(np.float64)
    low = group['low'].values.astype(np.float64)
    open = group['low'].values.astype(np.float64)
    
    group['atr10'] = talib.ATR(high, low, close, 10)
    group['ibs'] = _calculate_ibs(group)

    # group['ATR14'] = talib.ATR(high, low, close, 14)
    
   
    # (Breakouts)：由两根K线构成的突破
    # group = _calc_bo_2(group)
    group = _calc_big_bo(group)

   
    #
    group = _calc_label_1to1_RR(group)

    group.drop(['ibs','range', 'atr10','bar_dir', 'direction'], axis=1, inplace=True,errors='ignore')
    return group

def _calc_label_1to1_RR(df):
    '''
    1. 突破 (Breakouts)：由两根K线构成的突破，具有以下特征：
    a. 这两根K线中⾄少有⼀根的波幅（range）⼤于过去10根K线的平均波幅（10ATR）
    b. 第⼆根K线可能收盘也可能不收盘在第⼀根K线的同向突破点之外，但它必须与第⼀根K线的⽅
    向相同，才能被认定为突破和持续⾏情。换句话说，这根代表持续⾏情的K线（followthrough bar）并不⼀定需要收盘在突破点之外。突破K线之外，只要⽅向与突破K线⼀致，即
    可被视为有效的持续⾏情。
    c. 该K线必须是强劲的：⼀根强劲K线的IBS（Indicator Bar Strength，指标K线强度）对于看涨K
    线应≥69，对于看跌K线应≤31。IBS将在本⽂档的后续部分进⾏解释。
    d. ⼀根突破K线可以同时是⼀根反转K线。
    '''


    
    
    # df['min_next_20'] = df["low"].rolling(window=20, min_periods=1).max().shift(1)
    df['day_2_tp'] = _calc_days_to_cross(df,type='tp')
    df['day_2_istop'] = _calc_days_to_cross(df,type='istop')
    
    
    df['LABEL_BO'] = np.where(
        ~df['breakout_bar'] | ((df['day_2_tp'] == 999) & (df['day_2_istop'] == 999) ),              # 如果 breakout_bar 为 False
        1,                                # 赋值为 1
        np.where(                         # 否则（即 breakout_bar 为 True）
            df['day_2_tp'] < df['day_2_istop'], # 再比较两列大小
            2,                              # 如果 tp > istop，赋值为 2
            3                               # 否则，赋值为 3
        )
    )

    
    # print(df)
    
    # df.drop(['tp','istop','day_2_tp','day_2_istop'], axis=1, inplace=True)
    
    return df



def _calc_days_to_cross(df,type,window=20):
   
    n = len(df)
    
    # 创建偏移矩阵的索引
    # 这一步构建了一个下三角矩阵的变体，用于对齐未来数据
    valid_len = n - window
    if valid_len <= 0: 
        return np.full(n, np.nan)
    
    

    
    # 广播比较：(N, 20) >= (N, 1)
    # 结果是一个 (N, 20) 的布尔矩阵
    if type == "tp":
        high_prices = df['high'].to_numpy()
        future_prices_matrix = np.array([high_prices[i+1:i+1+window] for i in range(valid_len)])
        targets = df['tp'].to_numpy()
        # 获取对应的目标价向量
        current_targets_vector = targets[:valid_len]
        reached_matrix = future_prices_matrix >= current_targets_vector[:, np.newaxis]
    else:
        low_prices = df['low'].to_numpy()
        future_prices_matrix = np.array([low_prices[i+1:i+1+window] for i in range(valid_len)])
        targets = df['istop'].to_numpy()
        # 获取对应的目标价向量
        current_targets_vector = targets[:valid_len]
        reached_matrix = future_prices_matrix <= current_targets_vector[:, np.newaxis]
    
    # 计算每一行的第一个 True 的位置
    # np.argmax 在 axis=1 上操作
    days_matrix = np.argmax(reached_matrix, axis=1) + 1
    
    # 处理那些完全没有达到的行（全为 False 的行，argmax 会返回 0，需要修正为 NaN）
    days_matrix[~np.any(reached_matrix, axis=1)] = 999
    
    # 合并回完整长度的结果
    result = np.full(n, np.nan)
    result[:valid_len] = days_matrix
    
    result = np.nan_to_num(result, nan=999) # 将所有 NaN 替换为 999
    return result


def _calculate_ibs(df):

    """计算IBS (Indicator Bar Strength)"""
    
    # IBS = (收盘价 - 最低价) / (最高价 - 最低价) * 100
    # 避免除零错误
    range_ = df['high'] - df['low']
  
    ibs = (df['close'] - df['low']) / range_ * 100
    
    zero_range_mask = range_ == 0 
    # 今日close > 昨日close 的情况
    price_up_mask = df['close'] > df['close'].shift(1)
    ibs = np.where(zero_range_mask & price_up_mask, 100, ibs)
    # 今日close <= 昨日close 的情况
    ibs = np.where(zero_range_mask & ~price_up_mask, 0, ibs)
    
    return ibs


# 向量化版本（更高效）
def _calculate_direct(df):
    """
    向量化版本的K线方向分类（针对MultiIndex格式，性能更好）
    """

    # 初始化bar_dir为NaN
    df['bar_dir'] = np.nan
    
    # 对每个instrument分别处理
    instruments = df.index.get_level_values('instrument').unique()
    
    for instrument in instruments:
        # 获取当前instrument的数据
        instrument_mask = df.index.get_level_values('instrument') == instrument
        instrument_data = df.loc[instrument_mask]
        
        # 获取该instrument的索引
        instrument_idx = df.index[instrument_mask]
        
        # 按时间顺序排序
        sorted_idx = instrument_idx.sort_values()
        instrument_df = df.loc[sorted_idx]
        
        # 使用向量化条件判断
        C = instrument_df['close']
        O = instrument_df['open']
        IBS = instrument_df['ibs']
        
        # 条件1: C > O and IBS >= 50
        cond1 = (C > O) & (IBS >= 50)
        df.loc[sorted_idx[cond1], 'bar_dir'] = 1
        
        # 条件2: C < O and IBS <= 50
        cond2 = (C < O) & (IBS <= 50)
        df.loc[sorted_idx[cond2], 'bar_dir'] = -1
        
        # 条件3: C = O and IBS > 50
        cond3 = (C == O) & (IBS > 50)
        df.loc[sorted_idx[cond3], 'bar_dir'] = 1
        
        # 条件4: C = O and IBS < 50
        cond4 = (C == O) & (IBS < 50)
        df.loc[sorted_idx[cond4], 'bar_dir'] = -1
        
        # 条件5: C = O and IBS = 50
        cond5 = (C == O) & (IBS == 50)
        # 对于这些行，需要继承前一根K线的方向
        for idx in sorted_idx[cond5]:
            pos = list(sorted_idx).index(idx)
            if pos > 0:
                prev_idx = sorted_idx[pos-1]
                df.loc[idx, 'bar_dir'] = df.loc[prev_idx, 'bar_dir']
            else:
                df.loc[idx, 'bar_dir'] = 0
        
        # 条件6: C < O and IBS > 50
        cond6 = (C < O) & (IBS > 50)
        df.loc[sorted_idx[cond6], 'bar_dir'] = 1
        
        # 条件7: C > O and IBS < 50
        cond7 = (C > O) & (IBS < 50)
        df.loc[sorted_idx[cond7], 'bar_dir'] = -1
        
        # 条件8: IBS = 50
        cond8 = (IBS == 50) & (df.loc[sorted_idx, 'bar_dir'].isna())
        # 对于这些行，需要继承前一根K线的方向
        for idx in sorted_idx[cond8]:
            pos = list(sorted_idx).index(idx)
            if pos > 0:
                prev_idx = sorted_idx[pos-1]
                df.loc[idx, 'bar_dir'] = df.loc[prev_idx, 'bar_dir']
            else:
                df.loc[idx, 'bar_dir'] = 0
        
        # 对于剩余未处理的NaN值，继承前一根K线的方向
        for i, idx in enumerate(sorted_idx):
            if pd.isna(df.loc[idx, 'bar_dir']):
                if i > 0:
                    prev_idx = sorted_idx[i-1]
                    df.loc[idx, 'bar_dir'] = df.loc[prev_idx, 'bar_dir']
                else:
                    df.loc[idx, 'bar_dir'] = 0
    
    # 转换为整数类型
    df['direction'] = df['bar_dir'].astype(int)
    
   
    return df

def _calc_bo_2(df):
    """
    识别满足图片中条件的突破模式。
    
    参数:
        df: DataFrame，必须包含列: 'open', 'high', 'low', 'close'
        atr_period: 计算ATR的周期，默认为10
    
    返回:
        df: 添加了识别结果的原始DataFrame，新增列:
            'pattern_flag': 布尔值，当当前K线作为“第二根K线”满足所有条件时为True
    """
    # 2. 计算每根K线的波幅(Range)和方向
    df['range'] = df['high'] - df['low']
    
    df = _calculate_direct(df) 

    
    # 3. 条件判断（针对每一行作为潜在的“第二根K线”）
    # a. 波幅条件：当前K线或前一根K线中，至少一根的range > 前10根K线的平均波幅(ATR)
    condition_a = (df['range'] > df['atr10']) | (df['range'].shift(1) > df['atr10'].shift(1))
    
    # b. 方向条件：当前K线方向与前一根K线方向相同
    condition_b = df['direction'] == df['direction'].shift(1)
    
    # c. 强度条件：根据方向判断IBS
    #    对于看涨K线 (direction == 1): IBS >= 69
    #    对于看跌K线 (direction == -1): IBS <= 31
    condition_c = ((df['direction'] == 1) & (df['ibs'] >= 69)) | ((df['direction'] == -1) & (df['ibs'] <= 31))
    
    # 综合所有条件：当前K线作为“第二根K线”需同时满足a、b、c
    df['breakout_bar'] = condition_a & condition_b & condition_c
    df['tp'] = (2* df['close']) - (df['low'].shift(1))
    df['istop'] = df['low'].shift(1)
    
    # 可选：标记“第一根K线”（即前一根K线，当它被作为突破K线时）
    # df['breakout_bar'] = df['pattern_flag'].shift(-1)

    return df


def _calc_big_bo(df):
    
    # 2. 计算每根K线的波幅(Range)和方向
    df['range'] = df['high'] - df['low']
    

    # a. 波幅条件：一根大K线的波幅大于过去10根K线平均波幅的2倍。
    condition_a = df['range'] >  2* df['atr10']
    
    
    # b. 强度条件：根据方向判断IBS
    #    对于看涨K线 (direction == 1): IBS >= 69
    #    对于看跌K线 (direction == -1): IBS <= 31
    condition_b = df['ibs'] >= 69
    
    # 综合所有条件：当前K线作为“第二根K线”需同时满足a、b、c
    df['breakout_bar'] = condition_a & condition_b 
    

    df['tp'] = (2* df['close']) - df['low']
    df['istop'] = df['low']
    
    return df