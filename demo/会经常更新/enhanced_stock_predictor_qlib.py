#!/usr/bin/env python3
"""
增强股价预测器 - 基于Qlib数据源
Enhanced Stock Price Predictor - Based on Qlib Data Source

完全基于Qlib数据源，使用Alpha158因子进行股票预测分析
Fully based on Qlib data source, using Alpha158 factors for stock prediction
"""

import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import pickle
import time
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor
)
from sklearn.linear_model import (
    LinearRegression, 
    Ridge, 
    Lasso
)
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data import D

warnings.filterwarnings('ignore')

class EnhancedStockPredictorQlib:
    """增强股价预测器类 - 基于Qlib数据源"""
    
    def __init__(self):
        """初始化增强预测器"""
        self.models = {}
        self.handler = None
        self.dataset = None
        self.stock_code = None
        self.prediction_results = {}
        self.all_stocks = []
        self.qlib_data = {}
        self.cache_dir = os.path.expanduser("~/.qlib/predictor_cache")
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化Qlib
        self._init_qlib()
        
        # 加载股票池
        self._load_stock_universe()
        
        # 初始化模型
        self._init_models()
    
    def _init_qlib(self):
        """初始化Qlib环境"""
        try:
            # 使用最新数据路径
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
            print("✓ Qlib环境初始化成功")
            
            # 测试数据连接
            test_instruments = D.instruments("csi300")
            print("✓ Qlib数据连接测试成功")
            
        except Exception as e:
            print(f"❌ Qlib初始化失败: {e}")
            sys.exit(1)
    
    def _load_stock_universe(self):
        """加载股票池"""
        try:
            # 从CSI300指数获取股票列表
            instruments_file = os.path.expanduser("~/.qlib/qlib_data/cn_data/instruments/csi300.txt")
            
            if os.path.exists(instruments_file):
                with open(instruments_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            stock_code = parts[0]
                            start_date = parts[1]
                            end_date = parts[2]
                            
                            self.all_stocks.append({
                                'code': stock_code,
                                'name': self._get_stock_name(stock_code),
                                'industry': '',
                                'source': 'qlib',
                                'start': start_date,
                                'end': end_date
                            })
                
                print(f"✓ 从Qlib加载CSI300股票池: {len(self.all_stocks)} 只股票")
            else:
                # 降级到all.txt
                self._load_all_stocks()
            
            # 设置数据时间范围 - 使用最新数据
            self.data_start_date = '2023-01-01'  # 从2023年开始获取更多数据
            self.data_end_date = '2025-09-01'    # 到今天的数据
            print(f"✓ 数据时间范围: {self.data_start_date} 到 {self.data_end_date}")
                
        except Exception as e:
            print(f"❌ 加载股票池失败: {e}")
            sys.exit(1)
    
    def _load_all_stocks(self):
        """从all.txt加载所有股票"""
        try:
            all_stocks_file = os.path.expanduser("~/.qlib/qlib_data/cn_data/instruments/all.txt")
            
            with open(all_stocks_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        stock_code = parts[0]
                        start_date = parts[1]
                        end_date = parts[2]
                        
                        # 只选择主要的股票（过滤掉一些特殊股票）
                        if stock_code.startswith(('SH60', 'SZ00', 'SZ30')):
                            self.all_stocks.append({
                                'code': stock_code,
                                'name': self._get_stock_name(stock_code),
                                'industry': '',
                                'source': 'qlib',
                                'start': start_date,
                                'end': end_date
                            })
            
            print(f"✓ 从Qlib加载股票池: {len(self.all_stocks)} 只股票")
            
        except Exception as e:
            print(f"❌ 加载股票池失败: {e}")
            sys.exit(1)
    
    def _get_stock_name(self, stock_code):
        """获取股票名称（简化版，实际应用可以从其他数据源获取）"""
        # 这里可以扩展为从文件或其他数据源获取股票名称
        stock_names = {
            'SH600000': '浦发银行',
            'SH600036': '招商银行',
            'SH600519': '贵州茅台',
            'SZ000001': '平安银行',
            'SZ000002': '万科A',
            'SZ300059': '东方财富',
        }
        return stock_names.get(stock_code, f'股票{stock_code}')
    
    def _init_models(self):
        """初始化机器学习模型"""
        print("🤖 初始化机器学习模型...")
        
        # 核心模型集合
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=1000),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        print(f"✓ 初始化 {len(self.models)} 个机器学习模型")
    
    def get_qlib_data(self, stock_code, start_date, end_date):
        """从Qlib获取股票数据"""
        cache_key = f"{stock_code}_{start_date}_{end_date}"
        cache_file = os.path.join(self.cache_dir, f'{cache_key}.pkl')
        
        # 检查缓存
        if os.path.exists(cache_file):
            mod_time = os.path.getmtime(cache_file)
            if time.time() - mod_time < 3600:  # 1小时内的缓存有效
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass  # 缓存文件损坏，重新获取
        
        try:
            # 获取基础价格数据
            fields = ['$close', '$open', '$high', '$low', '$volume']
            data = D.features([stock_code], fields, start_time=start_date, end_time=end_date)
            
            if data is not None and not data.empty:
                # 重新整理索引，确保是单只股票的时间序列数据
                if isinstance(data.index, pd.MultiIndex):
                    # 如果是MultiIndex，选择这只股票的数据
                    data = data.xs(stock_code, level=0)
                
                # 缓存数据
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                
                return data
            else:
                return pd.DataFrame()
            
        except Exception as e:
            print(f"⚠ 获取Qlib数据失败: {e}")
            return pd.DataFrame()
    
    def search_stocks(self, query):
        """搜索股票"""
        query = query.strip().upper()
        matches = []
        
        for stock in self.all_stocks:
            code = stock['code']
            name = stock.get('name', '')
            
            if (code == query or 
                code.endswith(query) or 
                query in code or
                query in name):
                matches.append(stock)
                if len(matches) >= 10:
                    break
        
        return matches
    
    def get_stock_info(self, stock_code):
        """获取股票信息"""
        try:
            matches = self.search_stocks(stock_code)
            if not matches:
                print(f"❌ 未找到匹配的股票: {stock_code}")
                return False
            
            stock_info = matches[0]
            self.stock_code = stock_info['code']
            
            print(f"\n📊 股票信息: {self.stock_code}")
            print(f"股票名称: {stock_info.get('name', '未知')}")
            print(f"所属行业: {stock_info.get('industry', '未知')}")
            print(f"数据源: Qlib")
            
            # 获取最新价格信息
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
            
            qlib_data = self.get_qlib_data(self.stock_code, start_date, end_date)
            if not qlib_data.empty:
                latest_price = qlib_data['$close'].iloc[-1]
                if len(qlib_data) >= 2:
                    price_change = (qlib_data['$close'].iloc[-1] / qlib_data['$close'].iloc[-2] - 1) * 100
                else:
                    price_change = 0
                
                print(f"最新价格: ¥{latest_price:.2f}")
                print(f"日涨跌幅: {price_change:+.2f}%")
                print(f"最新数据: {qlib_data.index[-1].strftime('%Y-%m-%d')}")
                
                # 保存数据用于后续分析
                self.qlib_data[self.stock_code] = qlib_data
            else:
                print("⚠ 无法获取最新数据")
            
            return True
            
        except Exception as e:
            print(f"❌ 获取股票信息失败: {e}")
            return False
    
    def prepare_data(self):
        """准备数据用于模型训练"""
        try:
            print("\n🔄 准备数据...")
            
            # 使用Alpha158因子
            return self._prepare_alpha158_data()
                
        except Exception as e:
            print(f"❌ 数据准备失败: {e}")
            return False
    
    def _prepare_alpha158_data(self):
        """准备Alpha158数据"""
        try:
            # 使用较长的时间范围获取更多数据
            train_end_date = '2024-12-31'  # 训练数据到2024年底
            
            # 使用Alpha158处理器
            self.handler = Alpha158(
                instruments=[self.stock_code],
                start_time=self.data_start_date,
                end_time=self.data_end_date,
                freq='day'
            )
            
            # 获取完整数据
            full_data = self.handler.fetch()
            
            if full_data.empty or len(full_data) < 100:
                print(f"⚠ Alpha158数据不足: {len(full_data)} 条记录")
                return False
            
            print(f"✓ Alpha158数据获取成功: {full_data.shape}")
            
            # 分离特征和标签
            label_cols = [col for col in full_data.columns if 'LABEL' in str(col)]
            feature_cols = [col for col in full_data.columns if 'LABEL' not in str(col)]
            
            if not label_cols:
                print("❌ 未找到标签列")
                return False
            
            # 处理NaN值
            print("处理缺失值...")
            features = full_data[feature_cols].fillna(full_data[feature_cols].mean()).fillna(0)
            labels = full_data[label_cols[0]].fillna(full_data[label_cols[0]].median())
            
            # 重新组合数据
            processed_data = features.copy()
            processed_data['label'] = labels
            
            # 删除仍然包含NaN的行
            processed_data = processed_data.dropna()
            
            if len(processed_data) < 50:
                print(f"⚠ 处理后数据不足: {len(processed_data)} 条记录")
                return False
            
            # 按时间排序并分割数据
            processed_data = processed_data.sort_index()
            split_point = int(len(processed_data) * 0.8)
            
            train_data = processed_data.iloc[:split_point]
            test_data = processed_data.iloc[split_point:]
            
            print(f"✓ Alpha158数据准备完成")
            print(f"训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")
            print(f"特征数量: {len(feature_cols)}")
            
            # 保存到实例变量
            self.train_data = train_data
            self.test_data = test_data
            
            return True
            
        except Exception as e:
            print(f"❌ 准备Alpha158数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_models(self):
        """训练模型"""
        try:
            print("\n🤖 训练机器学习模型...")
            
            # 分离特征和标签
            label_col = 'label'
            feature_cols = [col for col in self.train_data.columns if col != 'label']
            
            X_train = self.train_data[feature_cols]
            y_train = self.train_data[label_col]
            X_test = self.test_data[feature_cols]
            y_test = self.test_data[label_col]
            
            # 最后检查NaN值
            print("最终数据检查...")
            assert not X_train.isna().any().any(), "训练特征仍包含NaN"
            assert not y_train.isna().any(), "训练标签仍包含NaN"
            assert not X_test.isna().any().any(), "测试特征仍包含NaN"
            assert not y_test.isna().any(), "测试标签仍包含NaN"
            
            # 数据标准化（部分模型需要）
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            print(f"数据形状: 训练集{X_train.shape}, 测试集{X_test.shape}")
            
            # 训练模型
            self.prediction_results = {}
            successful_models = 0
            
            for name, model in self.models.items():
                try:
                    print(f"训练 {name}...")
                    
                    # 选择合适的数据格式
                    if name == 'SVR':
                        model.fit(X_train_scaled, y_train)
                        test_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        test_pred = model.predict(X_test)
                    
                    # 评估模型
                    test_r2 = r2_score(y_test, test_pred)
                    test_mse = mean_squared_error(y_test, test_pred)
                    
                    self.prediction_results[name] = {
                        'model': model,
                        'scaler': scaler if name == 'SVR' else None,
                        'test_pred': test_pred,
                        'test_r2': test_r2,
                        'test_mse': test_mse
                    }
                    
                    successful_models += 1
                    
                except Exception as e:
                    print(f"  ⚠ {name} 训练失败: {e}")
            
            print(f"\n✓ 成功训练 {successful_models}/{len(self.models)} 个模型")
            
            if successful_models > 0:
                self._evaluate_models()
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ 模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _evaluate_models(self):
        """评估模型性能"""
        print("\n📈 模型性能评估:")
        print("-" * 60)
        
        # 按R²得分排序
        sorted_results = sorted(
            self.prediction_results.items(),
            key=lambda x: x[1]['test_r2'],
            reverse=True
        )
        
        print(f"{'模型':<15} {'测试R²':<10} {'测试MSE':<12}")
        print("-" * 40)
        
        for name, result in sorted_results:
            r2 = result['test_r2']
            mse = result['test_mse']
            print(f"{name:<15} {r2:<10.4f} {mse:<12.6f}")
        
        if sorted_results:
            best_model = sorted_results[0][0]
            best_r2 = sorted_results[0][1]['test_r2']
            print(f"\n🏆 最佳模型: {best_model} (R² = {best_r2:.4f})")
    
    def generate_prediction(self):
        """生成预测结果"""
        try:
            print(f"\n🎯 {self.stock_code} 预测分析:")
            print("=" * 60)
            
            if not self.prediction_results:
                print("⚠ 无预测结果")
                return
            
            # 集成预测
            predictions = {}
            weights = {}
            
            for name, result in self.prediction_results.items():
                pred = np.mean(result['test_pred'][-5:])  # 最近5个预测的平均值
                predictions[name] = pred
                weights[name] = max(0, result['test_r2'])  # 基于R²的权重
            
            # 归一化权重
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                weights = {k: 1/len(weights) for k in weights.keys()}
            
            # 加权预测
            weighted_pred = sum(pred * weights[name] for name, pred in predictions.items())
            
            print("\n📊 各模型预测:")
            for name, pred in predictions.items():
                weight = weights[name]
                trend = "上涨📈" if pred > 0 else "下跌📉"
                print(f"{name:<15}: {pred:+.4f} ({pred*100:+.2f}%) 权重:{weight:.3f} {trend}")
            
            print(f"\n🎯 集成预测结果:")
            print(f"预测收益率: {weighted_pred:+.4f} ({weighted_pred*100:+.2f}%)")
            print(f"预测趋势: {'上涨📈' if weighted_pred > 0 else '下跌📉'}")
            
            # 投资建议
            if weighted_pred > 0.02:
                suggestion = "强烈买入📈📈"
            elif weighted_pred > 0.005:
                suggestion = "买入📈"
            elif weighted_pred > -0.005:
                suggestion = "持有⚖️"
            else:
                suggestion = "观望📉"
            
            print(f"投资建议: {suggestion}")
            
            # 数据源说明
            print(f"\n📅 分析基于: Qlib Alpha158因子数据")
            print(f"数据时间: {self.data_start_date} - {self.data_end_date}")
            
            print(f"\n⚠️ 风险提示:")
            print(f"• 预测基于历史数据模式，存在不确定性")
            print(f"• 投资决策请结合基本面和市场环境") 
            print(f"• 投资有风险，入市需谨慎")
            
        except Exception as e:
            print(f"❌ 预测分析失败: {e}")
    
    def run_analysis(self):
        """运行分析"""
        print("🚀 增强股价预测器 - 基于Qlib数据源")
        print("=" * 70)
        
        print("✓ 使用Qlib数据 + Alpha158因子 + 机器学习模型")
        print(f"📊 支持股票: {len(self.all_stocks)} 只")
        print(f"🤖 ML模型: {len(self.models)} 个")
        
        while True:
            stock_code = input("\n请输入股票代码 (如: 000001, 600000) 或 'quit' 退出: ").strip()
            
            if stock_code.lower() == 'quit':
                print("感谢使用! 👋")
                break
            
            if not stock_code:
                print("⚠ 请输入有效股票代码")
                continue
            
            print(f"\n🔍 分析股票: {stock_code}")
            
            # 分析流程
            if (self.get_stock_info(stock_code) and 
                self.prepare_data() and 
                self.train_models()):
                
                self.generate_prediction()
                
                print(f"\n{'='*70}")
                print("✓ 分析完成!")
            else:
                print("❌ 分析失败，请重试")
            
            if input("\n继续分析其他股票? (y/n): ").lower() != 'y':
                print("感谢使用! 👋")
                break

def main():
    """主函数"""
    try:
        predictor = EnhancedStockPredictorQlib()
        predictor.run_analysis()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()