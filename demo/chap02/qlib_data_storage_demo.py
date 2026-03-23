import qlib
from qlib.data import D
import pandas as pd
import os
import time

def demo_storage_architecture():
    """演示Qlib数据存储架构"""
    print("=== Qlib数据存储架构演示 ===")
    
    # 初始化Qlib（请根据实际数据路径修改）
    qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")
    
    # 展示数据存储结构
    mount_path = os.path.expanduser("~/.qlib/qlib_data/cn_data")
    print(f"数据存储路径: {mount_path}")
    
    if os.path.exists(mount_path):
        print("\n数据目录结构:")
        for root, dirs, files in os.walk(mount_path):
            level = root.replace(mount_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # 只显示前3个文件
                print(f"{subindent}{file}")
            if len(files) > 3:
                print(f"{subindent}... 及其他 {len(files)-3} 个文件")
    
    # 展示二进制存储格式的优势
    print("\n=== 二进制存储性能对比 ===")
    
    # 获取测试数据
    data = D.features(
        instruments=['SH600000'],
        fields=['$close', '$volume', '$open', '$high', '$low'],
        start_time='2020-01-01',
        end_time='2020-01-31',
        freq='day'
    )
    
    print(f"数据形状: {data.shape}")
    print(f"内存使用: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # 演示快速数据访问
    start_time = time.time()
    sample_data = D.features(
        instruments=['SH600000', 'SH600036'],
        fields=['$close', '$volume'],
        start_time='2020-01-01',
        end_time='2020-01-10',
        freq='day'
    )
    end_time = time.time()
    
    print(f"数据访问时间: {(end_time - start_time) * 1000:.2f} ms")
    print(f"访问的数据量: {sample_data.shape}")

def demo_column_storage():
    """演示列式存储特点"""
    print("\n=== 列式存储特点演示 ===")
    
    # 获取多个特征的数据
    features = ['$close', '$volume', '$open', '$high', '$low']
    data = D.features(
        instruments=['SH600000'],
        fields=features,
        start_time='2020-01-01',
        end_time='2020-01-31',
        freq='day'
    )
    
    print("列式存储优势:")
    print("1. 按特征列存储，便于向量化计算")
    print("2. 支持高效的列操作")
    
    # 演示向量化计算
    print("\n向量化计算示例:")
    print(f"收盘价平均值: {data['$close'].mean():.2f}")
    print(f"成交量总和: {data['$volume'].sum():,.0f}")
    
    # 演示列相关性计算
    correlation = data[['$close', '$volume']].corr()
    print(f"\n价格与成交量相关性:")
    print(correlation)

def demo_index_optimization():
    """演示索引优化"""
    print("\n=== 索引优化演示 ===")
    
    # 时间索引优化
    print("1. 时间索引优化")
    calendar = D.calendar(start_time='2020-01-01', end_time='2020-01-10', freq='day')
    print(f"交易日历: {len(calendar)} 个交易日")
    print(f"日期范围: {calendar[0]} 到 {calendar[-1]}")
    
    # 股票代码索引
    print("\n2. 股票代码索引")
    instruments_conf = D.instruments('csi300')
    inst_list = D.list_instruments(instruments_conf, start_time="2020-01-01", end_time="2020-01-10", freq="day", as_list=True)
    print(f"CSI300成分股数量: {len(inst_list)}")
    print(f"样例股票代码: {inst_list[:5]}")
    
    # 快速数据定位
    print("\n3. 快速数据定位")
    start_time = time.time()
    specific_data = D.features(
        instruments=[inst_list[0]],
        fields=['$close'],
        start_time='2020-01-05',
        end_time='2020-01-05',
        freq='day'
    )
    end_time = time.time()
    
    print(f"单日单股票数据访问时间: {(end_time - start_time) * 1000:.2f} ms")
    print(f"数据内容: {specific_data}")

if __name__ == "__main__":
    try:
        demo_storage_architecture()
        demo_column_storage()
        demo_index_optimization()
        
        print("\n=== 存储格式总结 ===")
        print("Qlib数据存储优势:")
        print("- 二进制存储：高效读写，节省空间")
        print("- 列式存储：便于向量化计算")
        print("- 索引优化：支持快速数据定位")
        print("- 缓存机制：减少重复计算")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请确认:")
        print("1. Qlib数据已正确安装")
        print("2. 数据路径配置正确")
        print("3. 网络连接正常（如需下载数据）")