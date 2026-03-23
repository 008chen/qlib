#!/usr/bin/env python3
"""
最佳实践高级演示
基于第11章最佳实践与案例分析内容
包含：策略配置、错误处理、日志记录、Alpha158多因子模型、LSTM预测等
"""

import qlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import traceback
from functools import wraps
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.strategy.base import BaseStrategy
from qlib.data import D

warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

# === 错误处理和日志记录 ===

class TradingError(Exception):
    """交易相关错误基类"""
    pass

class DataError(TradingError):
    """数据相关错误"""
    pass

class StrategyError(TradingError):
    """策略相关错误"""
    pass

class RiskError(TradingError):
    """风险管理错误"""
    pass

def setup_logging(log_level=logging.INFO, log_file='trading_system.log'):
    """设置日志系统"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def error_handler(error_type=Exception):
    """错误处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                logger = logging.getLogger(func.__module__)
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        return wrapper
    return decorator

def log_performance(func):
    """性能日志装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger = logging.getLogger(func.__module__)
        
        logger.info(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Completed {func.__name__} in {duration:.2f} seconds")
            return result
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"Failed {func.__name__} after {duration:.2f} seconds: {str(e)}")
            raise
    return wrapper

# === 策略配置和基础框架 ===

@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    risk_degree: float = 0.95
    benchmark: str = "SH000300"
    universe: str = "csi300"
    topk: int = 50
    lookback_period: int = 20
    rebalance_frequency: int = 5
    max_position_size: float = 0.1
    stop_loss_threshold: float = -0.05
    take_profit_threshold: float = 0.1
    commission_rate: float = 0.0015

class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.positions = {}
        self.performance_history = []
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        pass
    
    def apply_risk_management(self, signals: pd.Series) -> pd.Series:
        """应用风险管理"""
        # 限制单个持仓规模
        max_weight = self.config.max_position_size
        signals = signals.clip(upper=max_weight)
        
        # 权重归一化
        total_weight = signals.sum()
        if total_weight > 1.0:
            signals = signals / total_weight
        
        return signals
    
    @log_performance
    def update_performance(self, returns: pd.Series):
        """更新性能记录"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'daily_return': returns.iloc[-1] if len(returns) > 0 else 0,
            'cumulative_return': returns.sum(),
            'volatility': returns.std(),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0
        })

class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    @error_handler(DataError)
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成动量信号"""
        if data.empty:
            raise DataError("输入数据为空")
        
        try:
            # 计算动量指标
            returns = data.pct_change(periods=self.config.lookback_period)
            momentum_scores = returns.iloc[-1]  # 最新的动量分数
            
            # 选择Top-K股票
            top_momentum = momentum_scores.nlargest(self.config.topk)
            
            # 生成等权重信号
            signals = pd.Series(
                index=top_momentum.index,
                data=1.0 / len(top_momentum)
            )
            
            # 应用风险管理
            signals = self.apply_risk_management(signals)
            
            self.logger.info(f"Generated {len(signals)} momentum signals")
            return signals
            
        except Exception as e:
            raise DataError(f"动量信号生成失败: {str(e)}")

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    @error_handler(DataError)
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成均值回归信号"""
        if data.empty:
            raise DataError("输入数据为空")
        
        try:
            # 计算移动平均和标准差
            ma = data.rolling(self.config.lookback_period).mean()
            std = data.rolling(self.config.lookback_period).std()
            
            # 计算Z-score
            current_prices = data.iloc[-1]
            current_ma = ma.iloc[-1]
            current_std = std.iloc[-1]
            
            z_scores = (current_prices - current_ma) / current_std
            
            # 选择偏离程度最大的股票（绝对值）
            top_deviations = z_scores.abs().nlargest(self.config.topk)
            
            # 生成反向信号
            signals = pd.Series(index=top_deviations.index, dtype=float)
            for stock in top_deviations.index:
                if z_scores[stock] > 2:  # 价格过高，做空
                    signals[stock] = -1.0 / len(top_deviations)
                elif z_scores[stock] < -2:  # 价格过低，做多
                    signals[stock] = 1.0 / len(top_deviations)
                else:
                    signals[stock] = 0
            
            # 只保留非零信号
            signals = signals[signals != 0]
            
            self.logger.info(f"Generated {len(signals)} mean reversion signals")
            return signals
            
        except Exception as e:
            raise DataError(f"均值回归信号生成失败: {str(e)}")

def create_strategy(strategy_type: str, config: StrategyConfig) -> BaseStrategy:
    """策略工厂函数"""
    strategy_map = {
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy
    }
    
    if strategy_type not in strategy_map:
        raise StrategyError(f"未知策略类型: {strategy_type}")
    
    return strategy_map[strategy_type](config)

# === Alpha158多因子模型 ===

class Alpha158MultiFactorModel:
    """Alpha158多因子模型"""
    
    def __init__(self, instruments='csi300', start_time='2018-01-01', end_time='2021-12-31'):
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.model = None
        self.factors = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @log_performance
    @error_handler(DataError)
    def prepare_data(self):
        """准备Alpha158数据"""
        try:
            handler = Alpha158(
                instruments=self.instruments,
                start_time=self.start_time,
                end_time=self.end_time,
                freq='day'
            )
            
            dataset = DatasetH(
                handler=handler,
                segments={
                    'train': ('2018-01-01', '2019-12-31'),
                    'valid': ('2020-01-01', '2020-06-30'),
                    'test': ('2020-07-01', '2021-12-31')
                }
            )
            
            self.dataset = dataset
            self.logger.info("Alpha158数据准备完成")
            
        except Exception as e:
            raise DataError(f"数据准备失败: {str(e)}")
    
    @log_performance
    def train_model(self):
        """训练LightGBM模型"""
        try:
            self.model = LGBModel(
                loss='mse',
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1
            )
            
            self.model.fit(self.dataset)
            self.logger.info("Alpha158模型训练完成")
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            raise
    
    def analyze_factors(self):
        """分析因子重要性"""
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'feature_importance'):
                # 获取特征重要性
                importance = self.model.model.feature_importance()
                feature_names = self.model.model.feature_name()
                
                # 创建重要性DataFrame
                factor_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                self.factors = factor_importance
                
                self.logger.info(f"分析了{len(factor_importance)}个因子的重要性")
                return factor_importance
            else:
                self.logger.warning("无法获取特征重要性")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"因子分析失败: {str(e)}")
            return pd.DataFrame()

class FactorAnalyzer:
    """因子分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_factor_ic(self, factor_data: pd.DataFrame, returns: pd.Series, periods: int = 1):
        """计算因子IC值"""
        try:
            ic_values = []
            
            for date in factor_data.index.get_level_values('datetime').unique():
                # 获取当日因子数据
                daily_factors = factor_data.loc[factor_data.index.get_level_values('datetime') == date]
                
                # 获取未来收益
                future_date = date + pd.Timedelta(days=periods)
                if future_date in returns.index.get_level_values('datetime'):
                    future_returns = returns.loc[returns.index.get_level_values('datetime') == future_date]
                    
                    # 计算相关系数
                    if len(daily_factors) > 1 and len(future_returns) > 1:
                        ic = daily_factors.corrwith(future_returns)
                        ic_values.append(ic)
            
            if ic_values:
                ic_df = pd.concat(ic_values, axis=1).T
                ic_mean = ic_df.mean()
                ic_std = ic_df.std()
                icir = ic_mean / ic_std
                
                self.logger.info(f"计算了{len(ic_values)}期的IC值")
                
                return {
                    'ic_mean': ic_mean,
                    'ic_std': ic_std,
                    'icir': icir,
                    'ic_series': ic_df
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"IC计算失败: {str(e)}")
            return {}

# === LSTM预测模型 ===

class LSTMPredictor(nn.Module):
    """LSTM预测模型"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # Dropout和全连接层
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output

class LSTMTrainingSystem:
    """LSTM训练系统"""
    
    def __init__(self, model, learning_rate=0.001, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.train_losses = []
        self.val_losses = []
    
    @log_performance
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    @log_performance
    def train(self, train_loader, val_loader, epochs=100, patience=10):
        """完整训练流程"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"开始训练LSTM模型，共{epochs}个epoch")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_lstm_model.pth'))
        self.logger.info("LSTM训练完成")
        
        return {
            'best_val_loss': best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

def create_lstm_data(data, sequence_length=20, target_column='LABEL0'):
    """创建LSTM训练数据"""
    try:
        # 分离特征和标签
        feature_cols = [col for col in data.columns if col != target_column]
        features = data[feature_cols].values
        targets = data[target_column].values
        
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"LSTM数据创建失败: {str(e)}")
        return np.array([]), np.array([])

# === 综合演示系统 ===

class ComprehensiveDemo:
    """综合演示系统"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.strategies = {}
        self.models = {}
        self.results = {}
    
    @log_performance
    def demo_strategy_framework(self):
        """演示策略框架"""
        self.logger.info("=== 策略框架演示 ===")
        
        # 创建策略配置
        config = StrategyConfig(
            name="DemoStrategy",
            topk=30,
            lookback_period=20,
            risk_degree=0.9
        )
        
        try:
            # 创建不同策略
            momentum_strategy = create_strategy('momentum', config)
            mean_reversion_strategy = create_strategy('mean_reversion', config)
            
            self.strategies['momentum'] = momentum_strategy
            self.strategies['mean_reversion'] = mean_reversion_strategy
            
            self.logger.info("策略创建完成")
            
            # 模拟数据测试
            np.random.seed(42)
            mock_data = pd.DataFrame(
                np.random.randn(100, 50) * 0.02 + 1,
                columns=[f'stock_{i}' for i in range(50)]
            ).cumprod()
            
            # 测试策略信号生成
            for name, strategy in self.strategies.items():
                try:
                    signals = strategy.generate_signals(mock_data)
                    self.logger.info(f"{name}策略生成了{len(signals)}个信号")
                    
                    # 模拟性能更新
                    mock_returns = pd.Series(np.random.randn(10) * 0.01)
                    strategy.update_performance(mock_returns)
                    
                except Exception as e:
                    self.logger.error(f"{name}策略测试失败: {str(e)}")
            
        except StrategyError as e:
            self.logger.error(f"策略框架演示失败: {str(e)}")
    
    @log_performance
    def demo_alpha158_model(self):
        """演示Alpha158多因子模型"""
        self.logger.info("=== Alpha158多因子模型演示 ===")
        
        try:
            # 创建Alpha158模型
            alpha158_model = Alpha158MultiFactorModel(
                instruments='csi300',
                start_time='2020-01-01',
                end_time='2020-12-31'
            )
            
            # 准备数据和训练模型
            alpha158_model.prepare_data()
            alpha158_model.train_model()
            
            # 分析因子
            factor_importance = alpha158_model.analyze_factors()
            
            if not factor_importance.empty:
                self.logger.info(f"Top 10 重要因子:")
                for i, (_, row) in enumerate(factor_importance.head(10).iterrows()):
                    self.logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            
            self.models['alpha158'] = alpha158_model
            
            # 因子分析
            analyzer = FactorAnalyzer()
            # 注意：这里简化了IC计算，实际需要真实的收益数据
            self.logger.info("Alpha158模型演示完成")
            
        except (DataError, Exception) as e:
            self.logger.error(f"Alpha158模型演示失败: {str(e)}")
    
    @log_performance
    def demo_lstm_model(self):
        """演示LSTM模型"""
        self.logger.info("=== LSTM模型演示 ===")
        
        try:
            # 创建模拟时序数据
            np.random.seed(42)
            sequence_length = 20
            input_size = 10
            batch_size = 32
            
            # 模拟训练数据
            n_samples = 1000
            X_train = np.random.randn(n_samples, sequence_length, input_size)
            y_train = np.random.randn(n_samples, 1)
            
            X_val = np.random.randn(200, sequence_length, input_size)
            y_val = np.random.randn(200, 1)
            
            # 转换为Tensor
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            
            # 创建DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # 创建LSTM模型
            lstm_model = LSTMPredictor(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                dropout=0.2
            )
            
            # 创建训练系统
            training_system = LSTMTrainingSystem(lstm_model, learning_rate=0.001)
            
            # 训练模型（使用较少的epoch进行演示）
            training_results = training_system.train(
                train_loader, val_loader, 
                epochs=5,  # 演示用较少epoch
                patience=3
            )
            
            self.models['lstm'] = lstm_model
            self.results['lstm_training'] = training_results
            
            self.logger.info(f"LSTM训练完成，最佳验证损失: {training_results['best_val_loss']:.6f}")
            
        except Exception as e:
            self.logger.error(f"LSTM模型演示失败: {str(e)}")
    
    def demo_error_handling(self):
        """演示错误处理"""
        self.logger.info("=== 错误处理演示 ===")
        
        # 演示数据错误处理
        try:
            empty_data = pd.DataFrame()
            strategy = self.strategies.get('momentum')
            if strategy:
                strategy.generate_signals(empty_data)
        except DataError as e:
            self.logger.info(f"成功捕获数据错误: {str(e)}")
        
        # 演示策略错误处理
        try:
            create_strategy('invalid_strategy', StrategyConfig('test'))
        except StrategyError as e:
            self.logger.info(f"成功捕获策略错误: {str(e)}")
        
        # 演示一般错误处理
        try:
            raise RiskError("演示风险管理错误")
        except RiskError as e:
            self.logger.info(f"成功捕获风险错误: {str(e)}")
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        self.logger.info("=== 生成综合报告 ===")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'strategies_tested': len(self.strategies),
            'models_trained': len(self.models),
            'results_summary': {}
        }
        
        # 策略测试结果
        for name, strategy in self.strategies.items():
            if strategy.performance_history:
                latest_perf = strategy.performance_history[-1]
                report['results_summary'][f'{name}_strategy'] = {
                    'cumulative_return': latest_perf['cumulative_return'],
                    'volatility': latest_perf['volatility'],
                    'sharpe_ratio': latest_perf['sharpe_ratio']
                }
        
        # LSTM训练结果
        if 'lstm_training' in self.results:
            report['results_summary']['lstm_model'] = {
                'best_val_loss': self.results['lstm_training']['best_val_loss'],
                'training_epochs': len(self.results['lstm_training']['train_losses'])
            }
        
        self.logger.info("综合报告生成完成")
        return report
    
    def run_comprehensive_demo(self):
        """运行综合演示"""
        self.logger.info("开始运行综合演示系统")
        
        try:
            # 演示各个组件
            self.demo_strategy_framework()
            self.demo_alpha158_model()
            self.demo_lstm_model()
            self.demo_error_handling()
            
            # 生成最终报告
            final_report = self.generate_comprehensive_report()
            
            self.logger.info("=== 演示完成 ===")
            self.logger.info(f"测试了{final_report['strategies_tested']}个策略")
            self.logger.info(f"训练了{final_report['models_trained']}个模型")
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"综合演示运行失败: {str(e)}")
            return None

def main():
    """主函数"""
    print("=== 最佳实践高级演示 ===")
    
    # 创建并运行综合演示
    demo = ComprehensiveDemo()
    
    try:
        final_report = demo.run_comprehensive_demo()
        
        if final_report:
            print("\n=== 演示结果摘要 ===")
            print(f"完成时间: {final_report['timestamp']}")
            print(f"策略测试数量: {final_report['strategies_tested']}")
            print(f"模型训练数量: {final_report['models_trained']}")
            
            print("\n=== 关键最佳实践 ===")
            print("1. 完善的错误处理和异常管理")
            print("2. 详细的日志记录和性能监控")
            print("3. 模块化的策略设计框架")
            print("4. 数据驱动的因子分析方法")
            print("5. 深度学习模型的规范化训练")
            print("6. 配置化的参数管理")
            print("7. 装饰器模式的功能扩展")
            
            print("\n=== 生产环境建议 ===")
            print("1. 实施更严格的数据验证")
            print("2. 增加更多的风险控制措施")
            print("3. 建立完善的监控和告警系统")
            print("4. 实现模型版本管理和A/B测试")
            print("5. 增加更多的性能优化策略")
            
        else:
            print("演示运行失败，请检查日志了解详细错误信息")
    
    except Exception as e:
        print(f"程序运行失败: {str(e)}")
    
    print("\n最佳实践高级演示完成！")
    print("\n主要演示内容:")
    print("1. 完整的错误处理体系")
    print("2. 专业的日志记录系统")
    print("3. 可配置的策略框架")
    print("4. Alpha158多因子模型最佳实践")
    print("5. LSTM深度学习模型标准化训练")
    print("6. 装饰器模式的优雅实现")
    print("7. 模块化设计和代码组织")

if __name__ == "__main__":
    main()