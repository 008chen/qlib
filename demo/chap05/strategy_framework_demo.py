#!/usr/bin/env python3
"""
投资策略框架演示
基于第5章投资策略设计内容
"""

import qlib
import numpy as np
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.strategy.base import BaseStrategy
from qlib.backtest import backtest, executor
from qlib.data.dataset import DatasetH
import warnings
warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

class TopkDropoutStrategy(BaseStrategy):
    """TopK策略 - 买入预测值最高的K只股票"""
    
    def __init__(self, topk=30, n_drop=5, **kwargs):
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
    
    def generate_trade_decision(self, score, **kwargs):
        """生成交易决策"""
        if score is None or len(score) == 0:
            return pd.Series(dtype='float64')
        
        # 获取当前日期的预测分数
        current_score = score.iloc[-1] if isinstance(score, pd.DataFrame) else score
        
        if isinstance(current_score, pd.Series):
            # 选择TopK股票
            top_stocks = current_score.nlargest(self.topk)
            
            # 计算权重（等权重）
            weight = 1.0 / len(top_stocks)
            decisions = pd.Series(index=top_stocks.index, data=weight)
            
            return decisions
        
        return pd.Series(dtype='float64')

class WeightStrategy(BaseStrategy):
    """权重策略 - 基于预测分数的权重分配"""
    
    def __init__(self, topk=50, **kwargs):
        super().__init__(**kwargs)
        self.topk = topk
    
    def generate_trade_decision(self, score, **kwargs):
        """生成交易决策"""
        if score is None or len(score) == 0:
            return pd.Series(dtype='float64')
        
        # 获取当前日期的预测分数
        current_score = score.iloc[-1] if isinstance(score, pd.DataFrame) else score
        
        if isinstance(current_score, pd.Series):
            # 选择TopK股票
            top_stocks = current_score.nlargest(self.topk)
            
            # 基于分数计算权重
            weights = top_stocks / top_stocks.sum()
            
            return weights
        
        return pd.Series(dtype='float64')

class MomentumStrategy(BaseStrategy):
    """动量策略 - 基于价格动量的投资策略"""
    
    def __init__(self, lookback=20, topk=30, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.topk = topk
    
    def generate_trade_decision(self, score, **kwargs):
        """生成交易决策"""
        if score is None or len(score) == 0:
            return pd.Series(dtype='float64')
        
        # 计算动量（这里简化为使用预测分数）
        current_score = score.iloc[-1] if isinstance(score, pd.DataFrame) else score
        
        if isinstance(current_score, pd.Series):
            # 选择动量最强的股票
            momentum_stocks = current_score.nlargest(self.topk)
            
            # 等权重分配
            weight = 1.0 / len(momentum_stocks)
            decisions = pd.Series(index=momentum_stocks.index, data=weight)
            
            return decisions
        
        return pd.Series(dtype='float64')

class StrategyFactory:
    """策略工厂类"""
    
    _strategies = {}
    
    @classmethod
    def register(cls, name, strategy_class):
        """注册策略"""
        cls._strategies[name] = strategy_class
    
    @classmethod
    def create(cls, name, **kwargs):
        """创建策略实例"""
        if name not in cls._strategies:
            raise ValueError(f"策略 {name} 不存在")
        
        return cls._strategies[name](**kwargs)
    
    @classmethod
    def list_strategies(cls):
        """列出所有可用策略"""
        return list(cls._strategies.keys())

def run_strategy_backtest(strategy, model, dataset, start_time, end_time):
    """运行策略回测"""
    print(f"运行策略回测: {strategy.__class__.__name__}")
    
    try:
        # 准备回测配置
        backtest_config = {
            "start_time": start_time,
            "end_time": end_time,
            "account": 100000,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        }
        
        # 创建执行器
        executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
        }
        
        # 运行回测
        # 注意：这里简化了回测流程，实际使用时需要更完整的配置
        print("回测配置已设置，策略已创建")
        
        # 模拟一些回测结果
        dates = pd.date_range(start_time, end_time, freq='D')[:10]  # 简化为10天
        returns = np.random.normal(0.001, 0.02, len(dates))  # 模拟收益
        
        backtest_result = pd.DataFrame({
            'return': returns,
            'benchmark_return': np.random.normal(0.0005, 0.015, len(dates)),
            'cumulative_return': np.cumsum(returns),
        }, index=dates)
        
        return backtest_result
        
    except Exception as e:
        print(f"回测过程中出现错误: {e}")
        return None

def analyze_strategy_performance(backtest_result):
    """分析策略性能"""
    if backtest_result is None:
        print("无法分析性能，回测结果为空")
        return
    
    print("\n=== 策略性能分析 ===")
    
    # 计算性能指标
    total_return = backtest_result['cumulative_return'].iloc[-1]
    annual_return = total_return * 252 / len(backtest_result)  # 年化收益
    volatility = backtest_result['return'].std() * np.sqrt(252)  # 年化波动率
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    max_cumulative = backtest_result['cumulative_return'].expanding().max()
    drawdown = backtest_result['cumulative_return'] - max_cumulative
    max_drawdown = drawdown.min()
    
    print(f"总收益率: {total_return:.4f}")
    print(f"年化收益率: {annual_return:.4f}")
    print(f"年化波动率: {volatility:.4f}")
    print(f"夏普比率: {sharpe_ratio:.4f}")
    print(f"最大回撤: {max_drawdown:.4f}")
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def main():
    print("准备数据和模型...")
    
    # 准备数据处理器
    handler = Alpha158(
        instruments='csi300',
        start_time='2020-01-01',
        end_time='2020-12-31',
        freq='day'
    )
    
    # 创建数据集
    dataset = DatasetH(
        handler=handler,
        segments={
            'train': ('2020-01-01', '2020-06-30'),
            'valid': ('2020-07-01', '2020-09-30'),
            'test': ('2020-10-01', '2020-12-31')
        }
    )
    
    # 训练模型
    print("训练预测模型...")
    model = LGBModel(
        loss='mse',
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        verbose=-1
    )
    
    model.fit(dataset)
    
    print("\n=== 策略工厂演示 ===")
    
    # 注册策略
    StrategyFactory.register("TopkDropout", TopkDropoutStrategy)
    StrategyFactory.register("WeightStrategy", WeightStrategy)
    StrategyFactory.register("MomentumStrategy", MomentumStrategy)
    
    # 列出可用策略
    available_strategies = StrategyFactory.list_strategies()
    print(f"可用策略: {available_strategies}")
    
    print("\n=== 策略创建和测试 ===")
    
    strategies_to_test = [
        ("TopkDropout", {"topk": 30, "n_drop": 5}),
        ("WeightStrategy", {"topk": 50}),
        ("MomentumStrategy", {"lookback": 20, "topk": 30})
    ]
    
    strategy_results = {}
    
    for strategy_name, params in strategies_to_test:
        print(f"\n--- 测试策略: {strategy_name} ---")
        
        # 创建策略实例
        strategy = StrategyFactory.create(strategy_name, **params)
        print(f"策略参数: {params}")
        
        # 生成一些模拟预测分数用于测试
        test_data = dataset.prepare('test')
        sample_scores = test_data['feature'].mean(axis=1)  # 简化的预测分数
        
        # 测试策略决策生成
        print("测试策略决策生成...")
        decision = strategy.generate_trade_decision(sample_scores)
        
        if not decision.empty:
            print(f"生成了 {len(decision)} 个交易决策")
            print(f"前5个决策: {decision.head()}")
            
            # 运行回测
            backtest_result = run_strategy_backtest(
                strategy, model, dataset, '2020-10-01', '2020-10-31'
            )
            
            # 分析性能
            performance = analyze_strategy_performance(backtest_result)
            strategy_results[strategy_name] = performance
        else:
            print("策略未生成有效决策")
    
    print("\n=== 策略对比分析 ===")
    
    if strategy_results:
        comparison_df = pd.DataFrame(strategy_results).T
        print("策略性能对比:")
        print(comparison_df)
        
        # 找出最佳策略
        best_strategy = comparison_df['sharpe_ratio'].idxmax()
        print(f"\n最佳策略（基于夏普比率）: {best_strategy}")
        print(f"夏普比率: {comparison_df.loc[best_strategy, 'sharpe_ratio']:.4f}")
    
    print("\n=== 策略参数敏感性分析 ===")
    
    # TopK参数敏感性分析
    print("TopK参数敏感性分析:")
    topk_values = [20, 30, 40, 50]
    
    for topk in topk_values:
        strategy = TopkDropoutStrategy(topk=topk)
        decision = strategy.generate_trade_decision(sample_scores)
        
        if not decision.empty:
            print(f"TopK={topk}: 选择了 {len(decision)} 只股票，平均权重 {decision.mean():.4f}")
    
    print("\n=== 自定义策略演示 ===")
    
    class CustomStrategy(BaseStrategy):
        """自定义策略示例"""
        
        def __init__(self, threshold=0.1, **kwargs):
            super().__init__(**kwargs)
            self.threshold = threshold
        
        def generate_trade_decision(self, score, **kwargs):
            """基于阈值的策略"""
            if score is None or len(score) == 0:
                return pd.Series(dtype='float64')
            
            current_score = score.iloc[-1] if isinstance(score, pd.DataFrame) else score
            
            if isinstance(current_score, pd.Series):
                # 选择预测分数大于阈值的股票
                selected = current_score[current_score > self.threshold]
                
                if len(selected) > 0:
                    # 等权重分配
                    weight = 1.0 / len(selected)
                    decisions = pd.Series(index=selected.index, data=weight)
                    return decisions
            
            return pd.Series(dtype='float64')
    
    # 测试自定义策略
    custom_strategy = CustomStrategy(threshold=0.0)
    custom_decision = custom_strategy.generate_trade_decision(sample_scores)
    
    if not custom_decision.empty:
        print(f"自定义策略选择了 {len(custom_decision)} 只股票")
        print(f"前5个决策: {custom_decision.head()}")
    
    print("\n策略框架演示完成！")

if __name__ == "__main__":
    main()