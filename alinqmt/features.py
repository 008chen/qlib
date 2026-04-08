import talib
import pandas as pd
import numpy as np




def calc_talib_features(group):
    '''
    单df
        
    '''
    group = group.dropna()
    close = group['$close'].values.astype(np.float64)
    high = group['$high'].values.astype(np.float64)
    low = group['$low'].values.astype(np.float64)
    open = group['$low'].values.astype(np.float64)
    
    group['atr10'] = talib.ATR(high, low, close, 10)
    group['ibs'] = _calculate_ibs(open, high, low, close)

    # group['ATR14'] = talib.ATR(high, low, close, 14)
    
    # print(group)
    group = _calc_breakout(group)
    # print(group)

    return group

def _calculate_ibs(open_, high, low, close):
    """计算IBS (Indicator Bar Strength)"""
    # IBS = (收盘价 - 最低价) / (最高价 - 最低价) * 100
    # 避免除零错误
    range_ = high - low
    range_[range_ == 0] = np.nan
    ibs = (close - low) / range_ * 100
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
        C = instrument_df['$close']
        O = instrument_df['$open']
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

def _calc_breakout(df):
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
    df['range'] = df['$high'] - df['$low']
    
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
    df['pattern_flag'] = condition_a & condition_b & condition_c
    
    # 可选：标记“第一根K线”（即前一根K线，当它被作为突破K线时）
    df['breakout_bar'] = df['pattern_flag'].shift(-1)
    
    
    df.drop(['ibs','range', '$close', '$high', '$low', '$open', '$volume','atr10','bar_dir', 'direction'], axis=1, inplace=True)
    
    return df