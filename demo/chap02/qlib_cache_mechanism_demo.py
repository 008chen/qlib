import qlib
from qlib.data import D
import pandas as pd
import time
import os
import tempfile
from qlib.config import REG_CN

def demo_cache_configuration():
    """演示缓存配置"""
    print("=== 缓存配置演示 ===")
    
    # 配置自定义缓存目录
    cache_dir = os.path.expanduser("~/.qlib/cache_demo")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"缓存目录: {cache_dir}")
    
    # 初始化Qlib时配置缓存
    qlib.init(
        mount_path="~/.qlib/qlib_data/cn_data",
        region=REG_CN,
        redis_host=None,  # 不使用Redis缓存
        redis_port=None,
        cache_dir=cache_dir,  # 设置缓存目录
        auto_mount=True
    )
    
    print("缓存配置完成")
    print("- 内存缓存: 启用")
    print("- 磁盘缓存: 启用")
    print(f"- 缓存路径: {cache_dir}")

def demo_memory_cache():
    """演示内存缓存"""
    print("\n=== 内存缓存演示 ===")
    
    # 第一次访问数据（无缓存）
    print("第一次数据访问（无缓存）:")
    start_time = time.time()
    data1 = D.features(
        instruments=['SH600000'],
        fields=['$close', '$volume', '$open', '$high', '$low'],
        start_time='2020-01-01',
        end_time='2020-01-31',
        freq='day'
    )
    first_access_time = time.time() - start_time
    print(f"访问时间: {first_access_time * 1000:.2f} ms")
    print(f"数据形状: {data1.shape}")
    
    # 第二次访问相同数据（内存缓存）
    print("\n第二次数据访问（内存缓存）:")
    start_time = time.time()
    data2 = D.features(
        instruments=['SH600000'],
        fields=['$close', '$volume', '$open', '$high', '$low'],
        start_time='2020-01-01',
        end_time='2020-01-31',
        freq='day'
    )
    second_access_time = time.time() - start_time
    print(f"访问时间: {second_access_time * 1000:.2f} ms")
    print(f"数据形状: {data2.shape}")
    
    # 性能提升计算
    if second_access_time > 0:
        speedup = first_access_time / second_access_time
        print(f"\n内存缓存性能提升: {speedup:.2f}x")
    
    # 验证数据一致性
    print(f"数据一致性检查: {'通过' if data1.equals(data2) else '失败'}")

def demo_disk_cache():
    """演示磁盘缓存"""
    print("\n=== 磁盘缓存演示 ===")
    
    cache_dir = os.path.expanduser("~/.qlib/cache_demo")
    
    # 清理内存缓存（重新初始化）
    qlib.init(
        mount_path="~/.qlib/qlib_data/cn_data",
        region=REG_CN,
        cache_dir=cache_dir,
        auto_mount=True
    )
    
    print("清理内存缓存后，测试磁盘缓存:")
    
    # 访问复杂计算的数据
    complex_fields = [
        '$close',
        'Mean($close, 20)',                # 移动平均
        'Std($close, 20)',                 # 标准差
        '$close / Ref($close, 1) - 1',     # 收益率
        'Corr($close, $volume, 20)',       # 相关性
    ]
    
    # 第一次访问复杂数据
    print("第一次访问复杂计算数据:")
    start_time = time.time()
    complex_data1 = D.features(
        instruments=['SH600000', 'SH600036'],
        fields=complex_fields,
        start_time='2020-01-01',
        end_time='2020-03-31',
        freq='day'
    )
    first_complex_time = time.time() - start_time
    print(f"访问时间: {first_complex_time * 1000:.2f} ms")
    print(f"数据形状: {complex_data1.shape}")
    
    # 检查缓存文件
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')][:3]
        print(f"缓存文件数量: {len([f for f in os.listdir(cache_dir) if f.endswith('.pkl')])}")
        print(f"缓存文件示例: {cache_files}")

def demo_cache_optimization():
    """演示缓存优化策略"""
    print("\n=== 缓存优化策略演示 ===")
    
    # 1. 预加载策略
    print("1. 数据预加载策略:")
    instruments = ['SH600000', 'SH600036', 'SH600519']
    base_fields = ['$close', '$volume', '$open', '$high', '$low']
    
    # 批量预加载数据
    preload_start = time.time()
    preloaded_data = {}
    for instrument in instruments:
        preloaded_data[instrument] = D.features(
            instruments=[instrument],
            fields=base_fields,
            start_time='2020-01-01',
            end_time='2020-12-31',
            freq='day'
        )
    preload_time = time.time() - preload_start
    print(f"预加载时间: {preload_time * 1000:.2f} ms")
    print(f"预加载股票数量: {len(instruments)}")
    
    # 2. 缓存命中率统计
    print("\n2. 缓存使用统计:")
    total_accesses = 0
    total_time = 0
    
    # 模拟多次随机访问
    import random
    for _ in range(5):
        instrument = random.choice(instruments)
        start_time = time.time()
        data = D.features(
            instruments=[instrument],
            fields=['$close', '$volume'],
            start_time='2020-06-01',
            end_time='2020-06-30',
            freq='day'
        )
        access_time = time.time() - start_time
        total_accesses += 1
        total_time += access_time
    
    average_time = total_time / total_accesses
    print(f"平均访问时间: {average_time * 1000:.2f} ms")
    print(f"总访问次数: {total_accesses}")

def demo_cache_management():
    """演示缓存管理"""
    print("\n=== 缓存管理演示 ===")
    
    cache_dir = os.path.expanduser("~/.qlib/cache_demo")
    
    if os.path.exists(cache_dir):
        # 计算缓存大小
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
                    file_count += 1
        
        print(f"缓存统计:")
        print(f"- 文件数量: {file_count}")
        print(f"- 总大小: {total_size / 1024 / 1024:.2f} MB")
        print(f"- 平均文件大小: {total_size / file_count / 1024:.2f} KB" if file_count > 0 else "- 平均文件大小: 0 KB")
        
        # 缓存清理建议
        print(f"\n缓存管理建议:")
        if total_size > 100 * 1024 * 1024:  # 100MB
            print("- 缓存较大，建议定期清理")
        else:
            print("- 缓存大小适中")
        
        print(f"- 缓存路径: {cache_dir}")
        print("- 可手动删除缓存文件以重置缓存")

def demo_cache_best_practices():
    """演示缓存最佳实践"""
    print("\n=== 缓存最佳实践 ===")
    
    print("缓存使用最佳实践:")
    
    # 1. 合理设置缓存目录
    print("1. 缓存目录设置:")
    print("   - 使用足够空间的磁盘")
    print("   - 避免与系统临时目录冲突")
    print("   - 定期清理过期缓存")
    
    # 2. 数据访问模式优化
    print("\n2. 数据访问模式:")
    print("   - 批量加载相关数据")
    print("   - 重复使用相同时间范围的数据")
    print("   - 避免频繁的小范围数据请求")
    
    # 演示批量vs单次访问的性能差异
    print("\n3. 批量访问性能对比:")
    
    # 单次访问
    single_start = time.time()
    date_ranges = [
        ('2020-01-01', '2020-01-31'),
        ('2020-02-01', '2020-02-29'),  # 修复：2020年2月只有29天
        ('2020-03-01', '2020-03-31')
    ]
    for start_date, end_date in date_ranges:
        single_data = D.features(
            instruments=['SH600000'],
            fields=['$close'],
            start_time=start_date,
            end_time=end_date,
            freq='day'
        )
    single_time = time.time() - single_start
    
    # 批量访问
    batch_start = time.time()
    batch_data = D.features(
        instruments=['SH600000'],
        fields=['$close'],
        start_time='2020-01-01',
        end_time='2020-03-31',
        freq='day'
    )
    batch_time = time.time() - batch_start
    
    print(f"   单次访问总时间: {single_time * 1000:.2f} ms")
    print(f"   批量访问时间: {batch_time * 1000:.2f} ms")
    print(f"   性能提升: {(single_time / batch_time):.2f}x")

if __name__ == "__main__":
    try:
        demo_cache_configuration()
        demo_memory_cache()
        demo_disk_cache()
        demo_cache_optimization()
        demo_cache_management()
        demo_cache_best_practices()
        
        print("\n=== 缓存机制总结 ===")
        print("Qlib缓存特性:")
        print("- 多级缓存: 内存 + 磁盘")
        print("- 自动管理: 透明缓存机制")
        print("- 性能优化: 显著提升数据访问速度")
        print("- 配置灵活: 支持自定义缓存策略")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("可能的原因:")
        print("1. Qlib数据未正确安装")
        print("2. 缓存目录权限不足")
        print("3. 磁盘空间不足")