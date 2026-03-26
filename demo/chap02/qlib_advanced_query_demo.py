import qlib
from qlib.data import D
import pandas as pd
import numpy as np

def demo_expression_support():
    """演示表达式支持"""
    print("=== 表达式支持演示 ===")
    
    # 初始化Qlib（请根据实际数据路径修改）
    qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")
    
    # 复杂表达式示例
    complex_fields = [
        '$close',                           # 收盘价
        'Ref($close, 1)',                  # 前一日收盘价  ，Reference的缩写
        'Mean($close, 20)',                # 20日移动平均
        '$close / Ref($close, 1) - 1',     # 日收益率
        'Std($close, 20)',                 # 20日标准差
        '($high + $low) / 2',              # 中位价
        '$volume / Mean($volume, 20)',      # 成交量比率
    ]
    
    print("支持的表达式类型:")
    for i, field in enumerate(complex_fields, 1):
        print(f"{i}. {field}")
    
    # 获取表达式数据
    data = D.features(
        instruments=['SH600000'],
        fields=complex_fields,  # 可以使用表达式或代码构建特征
        start_time='2020-01-01',
        end_time='2020-01-31',
        freq='day'
    )
    
    print(f"\n获取的数据形状: {data.shape}")
    print("数据样例:")
    print(data.head())

def demo_technical_indicators():
    """演示技术指标表达式"""
    print("\n=== 技术指标表达式演示 ===")
    
    # 技术指标表达式
    technical_fields = [
        '$close',
        'Mean($close, 5)',                 # 5日均线
        'Mean($close, 10)',                # 10日均线
        'Mean($close, 20)',                # 20日均线
        'Std($close, 20)',                 # 20日标准差
        'Max($high, 20)',                  # 20日最高价
        'Min($low, 20)',                   # 20日最低价
        'Rank($volume, 20)',               # 成交量排名
        'Corr($close, $volume, 20)',       # 价量相关性
    ]
    
    print("技术指标表达式:")
    for field in technical_fields:
        print(f"- {field}")
    
    # 获取技术指标数据
    tech_data = D.features(
        instruments=['SH600000'],
        fields=technical_fields,
        start_time='2020-01-01',
        end_time='2020-02-28',
        freq='day'
    )
    
    print(f"\n技术指标数据形状: {tech_data.shape}")
    
    # 计算一些派生指标
    print("\n派生技术指标:")
    
    # 布林带
    bb_upper = tech_data['Mean($close, 20)'] + 2 * tech_data['Std($close, 20)']
    bb_lower = tech_data['Mean($close, 20)'] - 2 * tech_data['Std($close, 20)']
    
    print(f"布林带上轨最新值: {bb_upper.iloc[-1]:.2f}")
    print(f"布林带下轨最新值: {bb_lower.iloc[-1]:.2f}")
    
    # 均线多头排列
    ma5_above_ma10 = tech_data['Mean($close, 5)'] > tech_data['Mean($close, 10)']
    ma10_above_ma20 = tech_data['Mean($close, 10)'] > tech_data['Mean($close, 20)']
    bullish_alignment = ma5_above_ma10 & ma10_above_ma20
    
    print(f"多头排列天数: {bullish_alignment.sum()} 天")

def demo_advanced_expressions():
    """演示高级表达式"""
    print("\n=== 高级表达式演示 ===")
    
    # 高级表达式示例
    advanced_fields = [
        '$close',
        # 相对位置指标
        '($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20))',
        
        # 动量指标
        '$close / Ref($close, 10) - 1',
        
        # 波动率调整收益
        '($close / Ref($close, 1) - 1) / Std($close / Ref($close, 1) - 1, 20)',
        
        # 量价背离指标
        'Corr($close / Ref($close, 1) - 1, $volume / Ref($volume, 1) - 1, 10)',
        
        # 趋势强度
        'Mean($close > Ref($close, 1), 20)',
    ]
    
    print("高级表达式:")
    for i, field in enumerate(advanced_fields, 1):
        if i > 1:  # 跳过基础价格
            print(f"{i-1}. {field}")
    
    # 获取高级表达式数据
    advanced_data = D.features(
        instruments=['SH600000'],
        fields=advanced_fields,
        start_time='2020-01-01',
        end_time='2020-03-31',
        freq='day'
    )
    
    print(f"\n高级表达式数据形状: {advanced_data.shape}")
    
    # 重命名列便于理解
    advanced_data.columns = [
        'close',
        'relative_position',
        'momentum_10d',
        'risk_adjusted_return',
        'price_volume_corr',
        'trend_strength'
    ]
    
    print("数据统计:")
    print(advanced_data.describe())

def demo_multi_instrument_expressions():
    """演示多股票表达式"""
    print("\n=== 多股票表达式演示 ===")
    
    # 获取多只股票数据
    instruments = ['SH600000', 'SH600036', 'SH600519']
    
    multi_fields = [
        '$close',
        'Mean($close, 20)',                # 移动平均
        '$volume / Mean($volume, 20)',      # 成交量比率
        '$close / Ref($close, 1) - 1',     # 收益率
    ]
    
    print(f"股票列表: {instruments}")
    print(f"表达式字段: {multi_fields}")
    
    # 获取多股票数据
    multi_data = D.features(
        instruments=instruments,
        fields=multi_fields,
        start_time='2020-01-01',
        end_time='2020-01-31',
        freq='day'
    )
    
    print(f"\n多股票数据形状: {multi_data.shape}")
    
    # 按股票分组分析
    for instrument in instruments:
        print(f"\n{instrument} 数据统计:")
        instrument_data = multi_data.xs(instrument, level='instrument')
        print(f"  平均收盘价: {instrument_data['$close'].mean():.2f}")
        print(f"  平均日收益率: {instrument_data['$close / Ref($close, 1) - 1'].mean():.4f}")
        print(f"  收益率标准差: {instrument_data['$close / Ref($close, 1) - 1'].std():.4f}")

def demo_conditional_filtering():
    """演示条件过滤"""
    print("\n=== 条件过滤演示 ===")
    
    # 基础数据获取
    base_data = D.features(
        instruments=['SH600000'],
        fields=['$close', '$volume', '$high', '$low'],
        start_time='2020-01-01',
        end_time='2020-03-31',
        freq='day'
    )
    
    print("条件过滤示例:")
    
    # 1. 大成交量交易日
    high_volume_mask = base_data['$volume'] > base_data['$volume'].quantile(0.8)
    high_volume_data = base_data[high_volume_mask]
    print(f"1. 高成交量交易日: {len(high_volume_data)} 天 (成交量 > 80%分位数)")
    
    # 2. 价格突破
    price_breakout_mask = base_data['$close'] > base_data['$close'].rolling(20).max().shift(1)
    breakout_data = base_data[price_breakout_mask]
    print(f"2. 价格突破交易日: {len(breakout_data)} 天 (收盘价 > 20日最高价)")
    
    # 3. 振幅过滤
    amplitude = (base_data['$high'] - base_data['$low']) / base_data['$low']
    high_amplitude_mask = amplitude > amplitude.quantile(0.9)
    high_amplitude_data = base_data[high_amplitude_mask]
    print(f"3. 高振幅交易日: {len(high_amplitude_data)} 天 (振幅 > 90%分位数)")
    
    # 4. 复合条件
    complex_condition = (
        (base_data['$volume'] > base_data['$volume'].quantile(0.7)) &
        (amplitude > amplitude.quantile(0.7)) &
        (base_data['$close'] > base_data['$close'].shift(1))
    )
    complex_filtered_data = base_data[complex_condition]
    print(f"4. 复合条件过滤: {len(complex_filtered_data)} 天")
    print("   条件: 高成交量 + 高振幅 + 价格上涨")

if __name__ == "__main__":
    try:
        demo_expression_support()
        demo_technical_indicators()
        demo_advanced_expressions()
        demo_multi_instrument_expressions()
        demo_conditional_filtering()
        
        print("\n=== 高级查询功能总结 ===")
        print("Qlib高级查询特性:")
        print("- 支持复杂数学表达式")
        print("- 内置技术指标函数")
        print("- 多股票并行计算")
        print("- 灵活的条件过滤")
        print("- 高效的向量化操作")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请确认:")
        print("1. Qlib数据已正确安装")
        print("2. 数据路径配置正确")
        print("3. 所需股票数据已下载")