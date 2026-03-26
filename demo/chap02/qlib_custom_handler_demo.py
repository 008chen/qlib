import qlib
from qlib.data import D
from qlib.contrib.data.handler import DataHandlerLP, Alpha158
from qlib.data.dataset import DatasetH
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Any
import warnings

class CustomDataHandler(Alpha158):
    """自定义数据处理器示例 - 继承Alpha158以简化实现"""
    
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        **kwargs
    ):
        """
        初始化自定义数据处理器
        
        继承Alpha158的稳定实现，只需重写特征配置
        """
        if learn_processors is None:
            learn_processors = [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ]
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            **kwargs
        )
    
    def get_feature_config(self):
        """使用Alpha158的配置格式但返回更少特征用于演示"""
        # 简化版本：使用Alpha158的基础配置但限制特征数量
        conf = {
            # "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],
            },
            # "rolling": {
            #     "windows": [5, 10, 20],
            #     "feature": ["ROC", "MA", "STD"],
            # },
        }
        from qlib.contrib.data.loader import Alpha158DL
        return Alpha158DL.get_feature_config(conf)
    
    def get_label_config(self):
        """重写标签配置"""
        return ["Ref($close, -1) / $close - 1"], ["LABEL0"] 
    
    def _get_custom_fields(self) -> List[str]:
        """定义自定义特征字段"""
        fields = [
            # 基础价格数据
            '$open', '$high', '$low', '$close', '$volume',
            
            # # 收益率特征
            # '$close / Ref($close, 1) - 1',      # 1日收益率
            # '$close / Ref($close, 5) - 1',      # 5日收益率
            # '$close / Ref($close, 10) - 1',     # 10日收益率
            # '$close / Ref($close, 20) - 1',     # 20日收益率
            
            # # 移动平均特征
            # 'Mean($close, 5)',                  # 5日均线
            # 'Mean($close, 10)',                 # 10日均线
            # 'Mean($close, 20)',                 # 20日均线
            # 'Mean($close, 60)',                 # 60日均线
            
            # # 波动率特征
            # 'Std($close, 5)',                   # 5日波动率
            # 'Std($close, 10)',                  # 10日波动率
            # 'Std($close, 20)',                  # 20日波动率
            
            # # 价格位置特征
            # '($close - Mean($close, 20)) / Std($close, 20)',  # 标准化价格位置
            # '($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20))',  # 相对位置
            
            # # 成交量特征
            # 'Mean($volume, 5)',                 # 5日平均成交量
            # 'Mean($volume, 20)',                # 20日平均成交量
            # '$volume / Mean($volume, 20)',      # 成交量比率
            # 'Std($volume, 20)',                 # 成交量波动率
            
            # 量价特征
            'Corr($close, $volume, 10)',        # 10日价量相关性
            # 'Corr($close / Ref($close, 1) - 1, $volume / Ref($volume, 1) - 1, 10)',  # 收益率与成交量变化相关性
            
            # # 趋势特征
            # 'Mean($close > Ref($close, 1), 10)', # 10日上涨概率
            # 'Mean($close > Mean($close, 20), 10)', # 10日均线上方概率
            
            # # 动量特征
            # '($close - Ref($close, 10)) / Ref($close, 10)',  # 10日动量
            # 'Max($close, 20) / Min($close, 20) - 1',         # 20日振幅
            
            # # 高级特征
            # 'RSI($close, 14)',                  # RSI指标（如果支持）
            # 'MACD($close)',                     # MACD指标（如果支持）
        ]
        
        # 过滤可能不支持的字段
        supported_fields = []
        for field in fields:
            if not any(indicator in field for indicator in ['RSI', 'MACD']):
                supported_fields.append(field)
        
        return supported_fields
    
    def get_feature_info(self) -> Dict[str, Any]:
        """获取特征配置信息（用于演示）"""
        fields = self._get_custom_fields()
        # print(fields)
        return {
            "feature_count": len(fields),
            "label_count": 1,
            "feature_groups": {
                "price": [f for f in fields if any(x in f for x in ['$open', '$high', '$low', '$close']) and not any(op in f for op in ['/', 'Mean', 'Std', 'Ref'])],
                "volume": [f for f in fields if '$volume' in f and not any(op in f for op in ['/', 'Mean', 'Std', 'Ref'])],
                "returns": [f for f in fields if '/ Ref($close' in f and '- 1' in f],
                "moving_avg": [f for f in fields if 'Mean($close' in f],
                "volatility": [f for f in fields if 'Std(' in f],
                "correlation": [f for f in fields if 'Corr(' in f],
                "trend": [f for f in fields if 'Mean($close >' in f],
            }
        }

class IndustryRotationHandler(Alpha158):
    """行业轮动数据处理器 - 继承Alpha158以简化实现"""
    
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        industry_field: str = "industry",
        **kwargs
    ):
        """
        行业轮动数据处理器初始化
        
        Parameters:
        -----------
        industry_field : 行业字段名
        """
        self.industry_field = industry_field
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            **kwargs
        )
    
    def get_feature_config(self):
        """行业轮动相关特征配置"""
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["CLOSE", "VWAP"],
            },
            "rolling": {
                "windows": [5, 20, 60],
                "feature": ["ROC", "MA"],
            },
        }
        from qlib.contrib.data.loader import Alpha158DL
        return Alpha158DL.get_feature_config(conf)
    
    def get_label_config(self):
        """5日收益率标签"""
        return ["Ref($close, -5) / $close - 1"], ["LABEL0"]

class FactorTimingHandler(Alpha158):
    """因子择时数据处理器 - 继承Alpha158以简化实现"""
    
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        **kwargs
    ):
        """因子择时数据处理器初始化"""
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            **kwargs
        )
    
    def get_feature_config(self):
        """择时相关特征配置"""
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["CLOSE", "OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {
                "windows": [5, 10, 20],
                "feature": ["ROC", "MA", "STD"],
            },
        }
        from qlib.contrib.data.loader import Alpha158DL
        return Alpha158DL.get_feature_config(conf)
    
    def get_label_config(self):
        """次日收益率标签"""
        return ["Ref($close, -1) / $close - 1"], ["LABEL0"]

def demo_custom_handler():
    """演示自定义数据处理器"""
    print("=== 自定义数据处理器演示 ===")
    
    # 初始化Qlib
    qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")
    
    # 创建自定义数据处理器
    custom_handler = CustomDataHandler(
        instruments='csi300',
        start_time='2020-01-01',
        end_time='2020-12-31',
        fit_start_time='2020-01-01',
        fit_end_time='2020-06-30',
        freq='day'
    )
    
    print("自定义数据处理器特点:")
    config = custom_handler.get_feature_info()
    print(f"- 特征总数: {config['feature_count']}")
    print(f"- 标签数量: {config['label_count']}")
    
    print(f"\n特征分组:")
    for group_name, features in config['feature_groups'].items():
        print(f"- {group_name}: {len(features)} 个特征")
    
    # 创建数据集
    dataset = DatasetH(
        handler=custom_handler,
        segments={
            'train': ('2020-01-01', '2020-06-30'),
            'valid': ('2020-07-01', '2020-09-30'),
            'test': ('2020-10-01', '2020-12-31')
        }
    )
    
    # 获取训练数据
    train_data = dataset.prepare('train')
    
    # 分离特征和标签
    label_cols = [col for col in train_data.columns if 'LABEL' in str(col)]
    feature_cols = [col for col in train_data.columns if 'LABEL' not in str(col)]
    
    features = train_data[feature_cols]
    labels = train_data[label_cols]
    
    print(f"\n训练数据形状:")
    print(f"- 特征: {features.shape}")
    print(f"- 标签: {labels.shape}")
    
    print(features)

def demo_industry_rotation_handler():
    """演示行业轮动处理器"""
    print("\n=== 行业轮动数据处理器演示 ===")
    
    try:
        # 创建行业轮动处理器
        industry_handler = IndustryRotationHandler(
            instruments=['SH600000', 'SH600036', 'SH600519'],
            start_time='2020-01-01',
            end_time='2020-12-31',
            fit_start_time='2020-01-01',
            fit_end_time='2020-06-30',
            freq='day'
        )
        
        print("行业轮动处理器特点:")
        print("- 基于Alpha158框架的行业轮动特征")
        print("- 关注行业相对强度")
        print("- 适用于行业配置策略")
        
        # 创建数据集
        industry_dataset = DatasetH(
            handler=industry_handler,
            segments={
                'train': ('2020-01-01', '2020-06-30'),
                'test': ('2020-07-01', '2020-12-31')
            }
        )
        
        train_data = industry_dataset.prepare('train')
        
        # 分离特征和标签
        label_cols = [col for col in train_data.columns if 'LABEL' in str(col)]
        feature_cols = [col for col in train_data.columns if 'LABEL' not in str(col)]
        
        features = train_data[feature_cols]
        labels = train_data[label_cols]
        
        print(f"\n行业轮动数据形状:")
        print(f"- 特征: {features.shape}")
        print(f"- 标签: {labels.shape}")
        
    except Exception as e:
        print(f"行业轮动处理器演示失败: {e}")

def demo_factor_timing_handler():
    """演示因子择时处理器"""
    print("\n=== 因子择时数据处理器演示 ===")
    
    try:
        # 创建因子择时处理器
        timing_handler = FactorTimingHandler(
            instruments=['SH600000'],
            start_time='2020-01-01',
            end_time='2020-12-31',
            fit_start_time='2020-01-01',
            fit_end_time='2020-06-30',
            freq='day'
        )
        
        print("因子择时处理器特点:")
        print("- 基于Alpha158框架的择时特征")
        print("- 关注市场时机选择")
        print("- 适用于择时策略")
        
        # 创建数据集
        timing_dataset = DatasetH(
            handler=timing_handler,
            segments={
                'train': ('2020-01-01', '2020-06-30'),
                'test': ('2020-07-01', '2020-12-31')
            }
        )
        
        train_data = timing_dataset.prepare('train')
        
        # 分离特征和标签
        label_cols = [col for col in train_data.columns if 'LABEL' in str(col)]
        feature_cols = [col for col in train_data.columns if 'LABEL' not in str(col)]
        
        features = train_data[feature_cols]
        labels = train_data[label_cols]
        
        print(f"\n择时数据形状:")
        print(f"- 特征: {features.shape}")
        print(f"- 标签: {labels.shape}")
        
        # 分析择时特征
        print(f"\n择时特征统计:")
        print(features.describe().iloc[:4, :5])
        
    except Exception as e:
        print(f"因子择时处理器演示失败: {e}")

def demo_handler_comparison():
    """演示处理器对比"""
    print("\n=== 数据处理器对比 ===")
    
    print("| 处理器类型     | 特征类型       | 适用场景       | 复杂度 |")
    print("|----------------|----------------|----------------|--------|")
    print("| CustomHandler  | 综合技术指标   | 通用量化策略   | 中等   |")
    print("| IndustryHandler| 行业相对特征   | 行业轮动策略   | 较高   |")
    print("| TimingHandler  | 择时信号特征   | 市场择时策略   | 中等   |")
    print("| Alpha158       | 经典特征集     | 传统机器学习   | 低     |")
    print("| Alpha360       | 丰富特征集     | 深度学习模型   | 高     |")
    
    print("\n选择建议:")
    print("1. 通用场景: 使用CustomHandler或Alpha158")
    print("2. 行业配置: 使用IndustryHandler")
    print("3. 择时策略: 使用TimingHandler")
    print("4. 深度学习: 使用Alpha360")
    print("5. 自定义需求: 继承DataHandlerLP构建专用处理器")

def demo_handler_best_practices():
    """演示处理器最佳实践"""
    print("\n=== 自定义处理器最佳实践 ===")
    
    print("设计原则:")
    print("1. 特征工程原则:")
    print("   - 业务逻辑驱动的特征设计")
    print("   - 避免过度拟合和数据泄露")
    print("   - 保持特征的可解释性")
    
    print("\n2. 数据处理原则:")
    print("   - 合理处理缺失值和异常值")
    print("   - 适当的数据标准化和归一化")
    print("   - 考虑前视偏差问题")
    
    print("\n3. 性能优化原则:")
    print("   - 平衡特征数量和计算效率")
    print("   - 利用Qlib的向量化计算")
    print("   - 合理使用缓存机制")
    
    print("\n4. 维护性原则:")
    print("   - 模块化的特征定义")
    print("   - 清晰的配置管理")
    print("   - 完善的文档和测试")

if __name__ == "__main__":
    try:
        demo_custom_handler()
        demo_industry_rotation_handler()
        demo_factor_timing_handler()
        demo_handler_comparison()
        demo_handler_best_practices()
        
        print("\n=== 自定义数据处理器总结 ===")
        print("关键要点:")
        print("1. 继承DataHandlerLP实现自定义处理器")
        print("2. 根据策略需求设计特征工程")
        print("3. 合理配置数据预处理流程")
        print("4. 考虑特征的业务含义和可解释性")
        print("5. 平衡模型复杂度和计算效率")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请检查:")
        print("1. Qlib是否正确安装和初始化")
        print("2. 数据处理器的依赖是否满足")
        print("3. 特征表达式是否支持")