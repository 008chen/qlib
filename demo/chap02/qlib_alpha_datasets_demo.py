import qlib
from qlib.data import D
from qlib.contrib.data.handler import Alpha158, Alpha360
from qlib.data.dataset import DatasetH
import pandas as pd
import numpy as np

def demo_alpha158_dataset():
    """演示Alpha158数据集"""
    print("=== Alpha158数据集演示 ===")
    
    # 初始化Qlib（请根据实际数据路径修改）
    qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")
    
    print("Alpha158数据集特点:")
    print("- 特征数量: 158个")
    print("- 数据频率: 日频")
    print("- 特征类型: 价格、成交量、技术指标")
    print("- 适用场景: 传统机器学习模型")
    
    # 创建Alpha158数据处理器
    handler = Alpha158(
        instruments='csi300',
        start_time='2020-01-01',
        end_time='2020-12-31',
        fit_start_time='2020-01-01',
        fit_end_time='2020-06-30',
        freq='day'
    )
    
    print(f"\n创建Alpha158处理器:")
    print(f"- 股票池: CSI300")
    print(f"- 时间范围: 2020-01-01 到 2020-12-31")
    print(f"- 拟合时间: 2020-01-01 到 2020-06-30")
    
    # 创建数据集
    dataset = DatasetH(
        handler=handler,
        segments={
            'train': ('2020-01-01', '2020-06-30'),
            'valid': ('2020-07-01', '2020-09-30'),
            'test': ('2020-10-01', '2020-12-31')
        }
    )
    
    print(f"\n数据集分割:")
    print(f"- 训练集: 2020-01-01 到 2020-06-30")
    print(f"- 验证集: 2020-07-01 到 2020-09-30")
    print(f"- 测试集: 2020-10-01 到 2020-12-31")
    
    # 获取数据样本
    train_data = dataset.prepare('train')
    
    # 分离特征和标签
    label_cols = [col for col in train_data.columns if 'LABEL' in str(col)]
    feature_cols = [col for col in train_data.columns if 'LABEL' not in str(col)]
    
    features = train_data[feature_cols]
    labels = train_data[label_cols]
    
    print(f"\n训练集数据形状:")
    print(f"- 特征: {features.shape}")
    print(f"- 标签: {labels.shape}")
    
    # 展示特征名称（前20个）
    feature_names = features.columns.tolist()
    print(f"\n特征示例（前20个）:")
    for i, name in enumerate(feature_names[:20], 1):
        print(f"{i:2d}. {name}")
    
    print(f"... 还有{len(feature_names) - 20}个特征")
    
    # 数据统计
    print(f"\n数据统计:")
    stats = features.describe()
    print(stats.iloc[:, :5])  # 显示前5个特征的统计信息

def demo_alpha360_dataset():
    """演示Alpha360数据集"""
    print("\n=== Alpha360数据集演示 ===")
    
    print("Alpha360数据集特点:")
    print("- 特征数量: 360个")
    print("- 数据频率: 日频")
    print("- 特征类型: 原始价格数据 + 技术指标")
    print("- 适用场景: 深度学习模型")
    
    # 创建Alpha360数据处理器
    try:
        handler360 = Alpha360(
            instruments='csi300',
            start_time='2020-01-01',
            end_time='2020-12-31',
            fit_start_time='2020-01-01',
            fit_end_time='2020-06-30',
            freq='day'
        )
        
        print(f"\n创建Alpha360处理器:")
        print(f"- 股票池: CSI300")
        print(f"- 时间范围: 2020-01-01 到 2020-12-31")
        
        # 创建数据集
        dataset360 = DatasetH(
            handler=handler360,
            segments={
                'train': ('2020-01-01', '2020-06-30'),
                'valid': ('2020-07-01', '2020-09-30'),
                'test': ('2020-10-01', '2020-12-31')
            }
        )
        
        # 获取数据样本
        train_data360 = dataset360.prepare('train')
        
        # 分离特征和标签
        label_cols360 = [col for col in train_data360.columns if 'LABEL' in str(col)]
        feature_cols360 = [col for col in train_data360.columns if 'LABEL' not in str(col)]
        
        features360 = train_data360[feature_cols360]
        labels360 = train_data360[label_cols360]
        
        print(f"\n训练集数据形状:")
        print(f"- 特征: {features360.shape}")
        print(f"- 标签: {labels360.shape}")
        
        # 特征分类分析
        feature_names360 = features360.columns.tolist()
        print(f"\n特征数量: {len(feature_names360)}")
        
        # 分析特征类型
        price_features = [f for f in feature_names360 if any(x in f for x in ['$close', '$open', '$high', '$low'])]
        volume_features = [f for f in feature_names360 if '$volume' in f]
        ma_features = [f for f in feature_names360 if 'Mean(' in f]
        std_features = [f for f in feature_names360 if 'Std(' in f]
        
        print(f"特征分类:")
        print(f"- 价格相关特征: {len(price_features)}")
        print(f"- 成交量相关特征: {len(volume_features)}")
        print(f"- 移动平均特征: {len(ma_features)}")
        print(f"- 标准差特征: {len(std_features)}")
        
    except Exception as e:
        print(f"Alpha360数据集加载失败: {e}")
        print("可能需要安装额外的依赖或数据")

def demo_feature_analysis():
    """演示特征分析"""
    print("\n=== 特征分析演示 ===")
    
    # 使用Alpha158进行特征分析
    handler = Alpha158(
        instruments='csi300',
        start_time='2020-01-01',
        end_time='2020-03-31',
        fit_start_time='2020-01-01',
        fit_end_time='2020-02-29',
        freq='day'
    )
    
    dataset = DatasetH(
        handler=handler,
        segments={
            'train': ('2020-01-01', '2020-02-29'),
            'test': ('2020-03-01', '2020-03-31')
        }
    )
    
    train_data = dataset.prepare('train')
    
    # 分离特征和标签
    label_cols = [col for col in train_data.columns if 'LABEL' in str(col)]
    feature_cols = [col for col in train_data.columns if 'LABEL' not in str(col)]
    
    features = train_data[feature_cols]
    labels = train_data[label_cols]
    
    print("特征质量分析:")
    
    # 1. 缺失值分析
    missing_ratio = features.isnull().sum() / len(features)
    high_missing_features = missing_ratio[missing_ratio > 0.1]
    print(f"1. 高缺失率特征 (>10%): {len(high_missing_features)}")
    if len(high_missing_features) > 0:
        print("   示例:")
        for name, ratio in high_missing_features.head().items():
            print(f"   - {name}: {ratio:.2%}")
    
    # 2. 特征相关性分析
    print(f"\n2. 特征与标签相关性分析:")
    correlations = features.corrwith(labels.iloc[:, 0])
    correlations = correlations.dropna()
    
    # 最相关的特征
    top_corr = correlations.abs().nlargest(10)
    print("   最相关特征 (Top 10):")
    for name, corr in top_corr.items():
        print(f"   - {name}: {correlations[name]:.4f}")
    
    # 3. 特征分布分析
    print(f"\n3. 特征分布分析:")
    numeric_features = features.select_dtypes(include=[np.number])
    skewness = numeric_features.skew().abs()
    high_skew_features = skewness[skewness > 2].nlargest(5)
    
    print("   高偏度特征 (Top 5):")
    for name, skew in high_skew_features.items():
        print(f"   - {name}: {skew:.2f}")

def demo_dataset_comparison():
    """演示数据集对比"""
    print("\n=== Alpha158 vs Alpha360 对比 ===")
    
    print("| 特性           | Alpha158      | Alpha360      |")
    print("|----------------|---------------|---------------|")
    print("| 特征数量       | 158           | 360           |")
    print("| 数据频率       | 日频          | 日频          |")
    print("| 计算复杂度     | 中等          | 较高          |")
    print("| 内存占用       | 较小          | 较大          |")
    print("| 适用模型       | 传统ML        | 深度学习      |")
    print("| 特征工程       | 经典指标      | 原始数据+指标 |")
    print("| 推荐场景       | 快速原型      | 精细化建模    |")
    
    print("\n使用建议:")
    print("1. 初学者/快速验证: 推荐Alpha158")
    print("2. 深度学习/精细建模: 推荐Alpha360")
    print("3. 计算资源有限: 选择Alpha158")
    print("4. 追求极致性能: 选择Alpha360")

def demo_custom_alpha_features():
    """演示自定义Alpha特征"""
    print("\n=== 自定义Alpha特征演示 ===")
    
    # 基于Alpha158框架构建自定义特征
    custom_fields = [
        # 基础价格特征
        '$close', '$open', '$high', '$low', '$volume',
        
        # 收益率特征
        '$close / Ref($close, 1) - 1',      # 1日收益率
        '$close / Ref($close, 5) - 1',      # 5日收益率
        '$close / Ref($close, 20) - 1',     # 20日收益率
        
        # 移动平均特征
        'Mean($close, 5)', 'Mean($close, 10)', 'Mean($close, 20)',
        
        # 波动率特征
        'Std($close, 5)', 'Std($close, 10)', 'Std($close, 20)',
        
        # 相对强弱特征
        '($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20))',
        
        # 量价特征
        'Corr($close, $volume, 10)',
        '$volume / Mean($volume, 20)',
        
        # 趋势特征
        'Mean($close > Ref($close, 1), 10)',  # 上涨概率
    ]
    
    print(f"自定义特征集 ({len(custom_fields)}个特征):")
    for i, field in enumerate(custom_fields, 1):
        print(f"{i:2d}. {field}")
    
    # 获取自定义特征数据
    try:
        custom_data = D.features(
            instruments=['SH600000', 'SH600036'],
            fields=custom_fields,
            start_time='2020-01-01',
            end_time='2020-01-31',
            freq='day'
        )
        
        print(f"\n自定义特征数据:")
        print(f"- 数据形状: {custom_data.shape}")
        print(f"- 特征数量: {len(custom_fields)}")
        
        # 特征统计
        print(f"\n特征统计摘要:")
        print(custom_data.describe().iloc[:4, :5])  # 显示前5个特征的统计
        
    except Exception as e:
        print(f"自定义特征获取失败: {e}")

if __name__ == "__main__":
    try:
        demo_alpha158_dataset()
        demo_alpha360_dataset()
        demo_feature_analysis()
        demo_dataset_comparison()
        demo_custom_alpha_features()
        
        print("\n=== Alpha数据集总结 ===")
        print("关键要点:")
        print("1. Alpha158: 经典特征集，适合传统机器学习")
        print("2. Alpha360: 丰富特征集，适合深度学习")
        print("3. 特征分析: 缺失值、相关性、分布检查")
        print("4. 自定义特征: 基于业务需求构建特征")
        print("5. 数据集选择: 根据模型和资源选择合适的数据集")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请检查:")
        print("1. Qlib数据是否正确安装")
        print("2. 数据路径是否配置正确")
        print("3. 所需的股票数据是否已下载")