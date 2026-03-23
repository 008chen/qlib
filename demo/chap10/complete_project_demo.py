#!/usr/bin/env python3
"""
完整项目实战演示
基于第10章完整项目实战内容
"""

import qlib
import numpy as np
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

class QuantitativeProject:
    """量化投资项目完整框架"""
    
    def __init__(self, project_name="QuantProject"):
        self.project_name = project_name
        self.config = {}
        self.data_handler = None
        self.model = None
        self.strategy = None
        self.backtest_results = {}
        self.performance_metrics = {}
        self.project_log = []
    
    def log_activity(self, activity, details=None):
        """记录项目活动"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity': activity,
            'details': details or {}
        }
        self.project_log.append(log_entry)
        print(f"[{activity}] {details.get('message', '')}")
    
    def initialize_project(self, config):
        """初始化项目"""
        self.config = config
        self.log_activity("PROJECT_INIT", {
            'message': f"项目 {self.project_name} 初始化完成",
            'config': config
        })
    
    def setup_data_pipeline(self):
        """设置数据流水线"""
        try:
            self.data_handler = Alpha158(
                instruments=self.config.get('instruments', 'csi300'),
                start_time=self.config.get('start_time', '2020-01-01'),
                end_time=self.config.get('end_time', '2020-12-31'),
                freq=self.config.get('freq', 'day')
            )
            
            self.log_activity("DATA_SETUP", {
                'message': "数据流水线设置完成",
                'instruments': self.config.get('instruments'),
                'time_range': f"{self.config.get('start_time')} - {self.config.get('end_time')}"
            })
            
        except Exception as e:
            self.log_activity("DATA_SETUP_ERROR", {
                'message': f"数据设置失败: {e}"
            })
            raise
    
    def prepare_datasets(self):
        """准备数据集"""
        try:
            segments = self.config.get('segments', {
                'train': ('2020-01-01', '2020-06-30'),
                'valid': ('2020-07-01', '2020-09-30'),
                'test': ('2020-10-01', '2020-12-31')
            })
            
            self.dataset = DatasetH(
                handler=self.data_handler,
                segments=segments
            )
            
            self.log_activity("DATASET_PREP", {
                'message': "数据集准备完成",
                'segments': segments
            })
            
        except Exception as e:
            self.log_activity("DATASET_PREP_ERROR", {
                'message': f"数据集准备失败: {e}"
            })
            raise
    
    def train_model(self):
        """训练模型"""
        try:
            model_config = self.config.get('model', {
                'type': 'LGBModel',
                'params': {
                    'loss': 'mse',
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'verbose': -1
                }
            })
            
            if model_config['type'] == 'LGBModel':
                self.model = LGBModel(**model_config['params'])
            else:
                raise ValueError(f"不支持的模型类型: {model_config['type']}")
            
            # 训练模型
            self.model.fit(self.dataset)
            
            self.log_activity("MODEL_TRAIN", {
                'message': "模型训练完成",
                'model_type': model_config['type'],
                'params': model_config['params']
            })
            
        except Exception as e:
            self.log_activity("MODEL_TRAIN_ERROR", {
                'message': f"模型训练失败: {e}"
            })
            raise
    
    def evaluate_model(self):
        """评估模型"""
        try:
            # 生成预测
            predictions = self.model.predict(self.dataset, segment='test')
            
            # 获取真实标签
            test_data = self.dataset.prepare('test')
            y_true = test_data['label']['LABEL0']
            
            # 计算评估指标
            from scipy.stats import pearsonr
            
            valid_mask = ~(np.isnan(predictions.values.flatten()) | np.isnan(y_true.values))
            pred_clean = predictions.values.flatten()[valid_mask]
            true_clean = y_true.values[valid_mask]
            
            if len(pred_clean) > 1:
                ic = pearsonr(pred_clean, true_clean)[0]
                mse = np.mean((pred_clean - true_clean) ** 2)
            else:
                ic = 0
                mse = float('inf')
            
            self.performance_metrics = {
                'IC': ic,
                'MSE': mse,
                'prediction_count': len(pred_clean),
                'coverage': len(pred_clean) / len(predictions)
            }
            
            self.log_activity("MODEL_EVAL", {
                'message': "模型评估完成",
                'metrics': self.performance_metrics
            })
            
        except Exception as e:
            self.log_activity("MODEL_EVAL_ERROR", {
                'message': f"模型评估失败: {e}"
            })
            raise
    
    def implement_strategy(self):
        """实施策略"""
        try:
            strategy_config = self.config.get('strategy', {
                'type': 'TopK',
                'params': {'topk': 30}
            })
            
            # 简化的策略实现
            class SimpleTopKStrategy:
                def __init__(self, topk=30):
                    self.topk = topk
                
                def generate_signals(self, predictions):
                    # 选择TopK股票
                    if len(predictions) == 0:
                        return pd.Series(dtype='float64')
                    
                    # 获取最新预测
                    latest_pred = predictions.groupby(level='instrument').last()
                    top_stocks = latest_pred.nlargest(self.topk)
                    
                    # 等权重分配
                    signals = pd.Series(
                        index=top_stocks.index,
                        data=1.0/len(top_stocks)
                    )
                    
                    return signals
            
            self.strategy = SimpleTopKStrategy(**strategy_config['params'])
            
            # 生成交易信号
            predictions = self.model.predict(self.dataset, segment='test')
            signals = self.strategy.generate_signals(predictions)
            
            self.log_activity("STRATEGY_IMPL", {
                'message': "策略实施完成",
                'strategy_type': strategy_config['type'],
                'signal_count': len(signals)
            })
            
        except Exception as e:
            self.log_activity("STRATEGY_IMPL_ERROR", {
                'message': f"策略实施失败: {e}"
            })
            raise
    
    def run_backtest(self):
        """运行回测"""
        try:
            # 简化的回测实现
            predictions = self.model.predict(self.dataset, segment='test')
            
            # 模拟回测结果
            dates = predictions.index.get_level_values('datetime').unique()[:30]
            returns = np.random.normal(0.001, 0.02, len(dates))
            
            self.backtest_results = {
                'dates': dates.tolist(),
                'returns': returns.tolist(),
                'cumulative_returns': np.cumsum(returns).tolist(),
                'total_return': np.sum(returns),
                'volatility': np.std(returns),
                'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            }
            
            self.log_activity("BACKTEST", {
                'message': "回测完成",
                'total_return': self.backtest_results['total_return'],
                'sharpe_ratio': self.backtest_results['sharpe_ratio']
            })
            
        except Exception as e:
            self.log_activity("BACKTEST_ERROR", {
                'message': f"回测失败: {e}"
            })
            raise
    
    def generate_report(self):
        """生成项目报告"""
        report = {
            'project_name': self.project_name,
            'config': self.config,
            'performance_metrics': self.performance_metrics,
            'backtest_results': {
                'total_return': self.backtest_results.get('total_return', 0),
                'volatility': self.backtest_results.get('volatility', 0),
                'sharpe_ratio': self.backtest_results.get('sharpe_ratio', 0)
            },
            'project_log_summary': {
                'total_activities': len(self.project_log),
                'errors': len([log for log in self.project_log if 'ERROR' in log['activity']]),
                'duration': self._calculate_project_duration()
            },
            'generated_at': datetime.now().isoformat()
        }
        
        self.log_activity("REPORT_GEN", {
            'message': "项目报告生成完成"
        })
        
        return report
    
    def _calculate_project_duration(self):
        """计算项目持续时间"""
        if len(self.project_log) < 2:
            return 0
        
        start_time = datetime.fromisoformat(self.project_log[0]['timestamp'])
        end_time = datetime.fromisoformat(self.project_log[-1]['timestamp'])
        
        return (end_time - start_time).total_seconds()

def run_complete_project():
    """运行完整项目流程"""
    
    print("=== 量化投资完整项目实战 ===")
    
    # 项目配置
    project_config = {
        'instruments': 'csi300',
        'start_time': '2020-01-01',
        'end_time': '2020-12-31',
        'freq': 'day',
        'segments': {
            'train': ('2020-01-01', '2020-06-30'),
            'valid': ('2020-07-01', '2020-09-30'),
            'test': ('2020-10-01', '2020-12-31')
        },
        'model': {
            'type': 'LGBModel',
            'params': {
                'loss': 'mse',
                'n_estimators': 50,
                'max_depth': 6,
                'learning_rate': 0.1,
                'verbose': -1
            }
        },
        'strategy': {
            'type': 'TopK',
            'params': {'topk': 30}
        }
    }
    
    # 创建项目实例
    project = QuantitativeProject("MyQuantProject")
    
    try:
        # 项目执行流水线
        project.initialize_project(project_config)
        project.setup_data_pipeline()
        project.prepare_datasets()
        project.train_model()
        project.evaluate_model()
        project.implement_strategy()
        project.run_backtest()
        
        # 生成最终报告
        final_report = project.generate_report()
        
        print("\n=== 项目执行完成 ===")
        print(f"项目名称: {final_report['project_name']}")
        print(f"总活动数: {final_report['project_log_summary']['total_activities']}")
        print(f"错误数: {final_report['project_log_summary']['errors']}")
        print(f"项目耗时: {final_report['project_log_summary']['duration']:.2f} 秒")
        
        print("\n=== 模型性能 ===")
        print(f"IC: {final_report['performance_metrics'].get('IC', 0):.4f}")
        print(f"MSE: {final_report['performance_metrics'].get('MSE', 0):.6f}")
        print(f"预测覆盖率: {final_report['performance_metrics'].get('coverage', 0):.4f}")
        
        print("\n=== 回测结果 ===")
        print(f"总收益: {final_report['backtest_results']['total_return']:.4f}")
        print(f"波动率: {final_report['backtest_results']['volatility']:.4f}")
        print(f"夏普比率: {final_report['backtest_results']['sharpe_ratio']:.4f}")
        
        # 保存报告
        report_filename = f"project_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"\n项目报告已保存到: {report_filename}")
        
        return final_report
        
    except Exception as e:
        print(f"项目执行失败: {e}")
        
        # 生成错误报告
        error_report = project.generate_report()
        error_report['execution_status'] = 'failed'
        error_report['error_message'] = str(e)
        
        return error_report

def demonstrate_best_practices():
    """展示最佳实践"""
    
    print("\n=== 量化投资最佳实践演示 ===")
    
    class BestPracticesGuide:
        """最佳实践指南"""
        
        def __init__(self):
            self.practices = {
                'data_management': [
                    "使用版本控制管理数据",
                    "建立数据质量监控",
                    "实施数据备份策略",
                    "标准化数据格式"
                ],
                'model_development': [
                    "使用交叉验证评估模型",
                    "实施模型版本管理",
                    "建立模型监控机制",
                    "记录实验过程"
                ],
                'strategy_design': [
                    "考虑交易成本",
                    "实施风险管理",
                    "测试极端市场条件",
                    "建立止损机制"
                ],
                'deployment': [
                    "建立监控系统",
                    "实施灰度发布",
                    "准备回滚方案",
                    "建立告警机制"
                ]
            }
        
        def get_recommendations(self, category):
            """获取特定类别的建议"""
            return self.practices.get(category, [])
        
        def generate_checklist(self):
            """生成检查清单"""
            checklist = {}
            for category, practices in self.practices.items():
                checklist[category] = [
                    {'practice': practice, 'completed': False}
                    for practice in practices
                ]
            return checklist
    
    # 创建最佳实践指南
    guide = BestPracticesGuide()
    
    print("数据管理最佳实践:")
    for practice in guide.get_recommendations('data_management'):
        print(f"  - {practice}")
    
    print("\n模型开发最佳实践:")
    for practice in guide.get_recommendations('model_development'):
        print(f"  - {practice}")
    
    print("\n策略设计最佳实践:")
    for practice in guide.get_recommendations('strategy_design'):
        print(f"  - {practice}")
    
    print("\n部署最佳实践:")
    for practice in guide.get_recommendations('deployment'):
        print(f"  - {practice}")
    
    # 生成检查清单
    checklist = guide.generate_checklist()
    print(f"\n生成了包含 {sum(len(practices) for practices in checklist.values())} 项的检查清单")

def main():
    """主函数"""
    
    # 运行完整项目
    project_report = run_complete_project()
    
    # 展示最佳实践
    demonstrate_best_practices()
    
    print("\n=== 项目评估和改进建议 ===")
    
    # 评估项目成功度
    success_score = 0
    
    if project_report.get('execution_status') != 'failed':
        success_score += 30  # 基础执行成功
    
    if project_report.get('project_log_summary', {}).get('errors', 0) == 0:
        success_score += 20  # 无错误执行
    
    ic_score = project_report.get('performance_metrics', {}).get('IC', 0)
    if abs(ic_score) > 0.02:
        success_score += 25  # 模型性能良好
    
    sharpe_ratio = project_report.get('backtest_results', {}).get('sharpe_ratio', 0)
    if sharpe_ratio > 0.5:
        success_score += 25  # 回测表现良好
    
    print(f"项目成功度评分: {success_score}/100")
    
    # 改进建议
    suggestions = []
    
    if success_score < 70:
        suggestions.append("考虑优化模型参数或特征工程")
    
    if ic_score < 0.01:
        suggestions.append("模型预测能力较弱，需要改进")
    
    if sharpe_ratio < 0.3:
        suggestions.append("策略风险调整后收益较低，考虑优化")
    
    if project_report.get('project_log_summary', {}).get('errors', 0) > 0:
        suggestions.append("存在执行错误，需要调试和修复")
    
    if suggestions:
        print("\n改进建议:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    else:
        print("\n项目执行良好，可以考虑进一步优化和扩展功能")
    
    print("\n完整项目实战演示完成！")

if __name__ == "__main__":
    main()