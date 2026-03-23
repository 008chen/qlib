#!/usr/bin/env python3
"""
增强股价预测器 - 集成Tushare实时数据源
Enhanced Stock Price Predictor - Integrated with Tushare Real-time Data

集成Tushare数据源，获取最新股票数据，结合Alpha158因子进行预测
Integrates Tushare data source for latest stock data with Alpha158 factors
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
import tushare as ts
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

warnings.filterwarnings('ignore')

class EnhancedStockPredictor:
    """增强股价预测器类 - 集成Tushare数据源"""
    
    def __init__(self):
        """初始化增强预测器"""
        self.models = {}
        self.handler = None
        self.dataset = None
        self.stock_code = None
        self.prediction_results = {}
        self.all_stocks = []
        self.tushare_data = {}
        self.cache_dir = os.path.expanduser("~/.qlib/tushare_cache")
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化Tushare
        self._init_tushare()
        
        # 初始化Qlib (用于Alpha158计算)
        self._init_qlib()
        
        # 加载股票池
        self._load_stock_universe()
        
        # 初始化模型
        self._init_models()
    
    def _init_tushare(self):
        """初始化Tushare数据源"""
        try:
            # 设置Tushare token
            ts.set_token('93aaf3f9c716f82074471a745442831bf8c415ab234136dcfe6d7bbc')
            self.pro = ts.pro_api()
            
            # 测试连接
            trade_cal = self.pro.trade_cal(exchange='', start_date='20240801', end_date='20240810')
            if not trade_cal.empty:
                print("✓ Tushare数据源连接成功")
                self.use_tushare = True
            else:
                raise Exception("连接测试失败")
                
        except Exception as e:
            print(f"⚠ Tushare连接失败: {e}")
            print("⚠ 将仅使用Qlib历史数据")
            self.use_tushare = False
            self.pro = None
    
    def _init_qlib(self):
        """初始化Qlib环境"""
        try:
            # 使用最新数据路径
            data_paths = ["~/.qlib/qlib_data/cn_data_new", "~/.qlib/qlib_data/cn_data"]
            
            for path in data_paths:
                if os.path.exists(os.path.expanduser(path)):
                    qlib.init(provider_uri=path, region=REG_CN)
                    print(f"✓ Qlib环境初始化成功 (数据路径: {path})")
                    return
                    
            raise Exception("未找到Qlib数据目录")
        except Exception as e:
            print(f"❌ Qlib初始化失败: {e}")
            sys.exit(1)
    
    def _load_stock_universe(self):
        """加载股票池"""
        try:
            if self.use_tushare:
                # 从Tushare获取最新股票列表
                stocks_df = self._get_tushare_stock_list()
                if not stocks_df.empty:
                    for _, row in stocks_df.iterrows():
                        ts_code = row['ts_code']
                        # 转换为qlib格式: 000001.SZ -> SZ000001
                        if '.SH' in ts_code:
                            qlib_code = 'SH' + ts_code.split('.')[0]
                        elif '.SZ' in ts_code:
                            qlib_code = 'SZ' + ts_code.split('.')[0]
                        else:
                            continue
                        
                        self.all_stocks.append({
                            'code': qlib_code,
                            'ts_code': ts_code,
                            'name': row.get('name', ''),
                            'industry': row.get('industry', ''),
                            'source': 'tushare'
                        })
                    
                    print(f"✓ 从Tushare加载股票池: {len(self.all_stocks)} 只股票")
                else:
                    raise Exception("Tushare股票列表为空")
            else:
                # 使用Qlib数据
                self._load_qlib_stock_list()
                
            # 设置数据时间范围
            if self.use_tushare:
                self.data_start_date = '2022-01-01'  # 从2022年开始
                self.data_end_date = datetime.now().strftime('%Y-%m-%d')
                print(f"✓ 数据时间范围: {self.data_start_date} 到 {self.data_end_date} (实时数据)")
            else:
                self.data_start_date = '2019-01-01'
                self.data_end_date = '2020-09-25'
                print(f"✓ 数据时间范围: {self.data_start_date} 到 {self.data_end_date} (历史数据)")
                
        except Exception as e:
            print(f"❌ 加载股票池失败: {e}")
            # 降级到Qlib数据
            if self.use_tushare:
                print("⚠ 降级使用Qlib历史数据")
                self.use_tushare = False
                self._load_qlib_stock_list()
    
    def _get_tushare_stock_list(self):
        """从Tushare获取股票列表"""
        cache_file = os.path.join(self.cache_dir, 'stock_list.pkl')
        
        # 检查缓存
        if os.path.exists(cache_file):
            mod_time = os.path.getmtime(cache_file)
            if time.time() - mod_time < 24 * 3600:  # 24小时内的缓存有效
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        try:
            # 获取股票基础信息
            stocks_df = self.pro.stock_basic(
                exchange='', 
                list_status='L', 
                fields='ts_code,symbol,name,area,industry,list_date'
            )
            
            # 缓存数据
            with open(cache_file, 'wb') as f:
                pickle.dump(stocks_df, f)
            
            return stocks_df
        except Exception as e:
            print(f"⚠ 获取Tushare股票列表失败: {e}")
            return pd.DataFrame()
    
    def _load_qlib_stock_list(self):
        """从Qlib加载股票列表"""
        try:
            all_stocks_file = os.path.expanduser("~/.qlib/qlib_data/cn_data/instruments/all.txt")
            if not os.path.exists(all_stocks_file):
                all_stocks_file = os.path.expanduser("~/.qlib/qlib_data/cn_data_new/instruments/all.txt")
            
            with open(all_stocks_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        stock_code = parts[0]
                        start_date = parts[1]
                        end_date = parts[2]
                        
                        self.all_stocks.append({
                            'code': stock_code,
                            'ts_code': '',
                            'name': '',
                            'industry': '',
                            'source': 'qlib',
                            'start': start_date,
                            'end': end_date
                        })
            
            print(f"✓ 从Qlib加载股票池: {len(self.all_stocks)} 只股票")
            
        except Exception as e:
            print(f"❌ 加载Qlib股票池失败: {e}")
            sys.exit(1)
    
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
    
    def get_tushare_data(self, ts_code, start_date, end_date):
        """从Tushare获取股票数据"""
        if not self.use_tushare or not self.pro:
            return pd.DataFrame()
        
        cache_key = f"{ts_code}_{start_date}_{end_date}"
        cache_file = os.path.join(self.cache_dir, f'{cache_key}.pkl')
        
        # 检查缓存
        if os.path.exists(cache_file):
            mod_time = os.path.getmtime(cache_file)
            if time.time() - mod_time < 3600:  # 1小时内的缓存有效
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        try:
            # 获取日线数据
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                fields='ts_code,trade_date,open,high,low,close,vol,amount'
            )
            
            if not df.empty:
                # 数据预处理
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df.set_index('trade_date', inplace=True)
                
                # 重命名列以匹配Qlib格式
                df.rename(columns={
                    'open': '$open',
                    'high': '$high', 
                    'low': '$low',
                    'close': '$close',
                    'vol': '$volume',
                    'amount': '$amount'
                }, inplace=True)
                
                # 删除ts_code列避免重复
                if 'ts_code' in df.columns:
                    df.drop('ts_code', axis=1, inplace=True)
                
                # 缓存数据
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
            
            return df
            
        except Exception as e:
            print(f"⚠ 获取Tushare数据失败: {e}")
            return pd.DataFrame()
    
    def search_stocks(self, query):
        """搜索股票"""
        query = query.strip().upper()
        matches = []
        
        for stock in self.all_stocks:
            code = stock['code']
            name = stock.get('name', '')
            ts_code = stock.get('ts_code', '')
            
            if (code == query or 
                code.endswith(query) or 
                ts_code.startswith(query) or
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
            ts_code = stock_info.get('ts_code', '')
            
            print(f"\n📊 股票信息: {self.stock_code}")
            print(f"股票名称: {stock_info.get('name', '未知')}")
            print(f"所属行业: {stock_info.get('industry', '未知')}")
            print(f"数据源: {stock_info.get('source', 'unknown')}")
            
            # 获取最新价格信息
            if self.use_tushare and ts_code:
                # 从Tushare获取最新数据
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
                
                tushare_data = self.get_tushare_data(ts_code, start_date, end_date)
                if not tushare_data.empty:
                    latest_price = tushare_data['$close'].iloc[-1]
                    if len(tushare_data) >= 2:
                        price_change = (tushare_data['$close'].iloc[-1] / tushare_data['$close'].iloc[-2] - 1) * 100
                    else:
                        price_change = 0
                    
                    print(f"最新价格: ¥{latest_price:.2f}")
                    print(f"日涨跌幅: {price_change:+.2f}%")
                    print(f"最新数据: {tushare_data.index[-1].strftime('%Y-%m-%d')}")
                    
                    # 保存数据用于后续分析
                    self.tushare_data[self.stock_code] = {
                        'ts_code': ts_code,
                        'data': tushare_data
                    }
                else:
                    print("⚠ 无法获取Tushare最新数据，将使用历史数据")
            else:
                print("使用历史数据进行分析")
            
            return True
            
        except Exception as e:
            print(f"❌ 获取股票信息失败: {e}")
            return False
    
    def prepare_data(self):
        """准备数据用于模型训练"""
        try:
            print("\n🔄 准备数据...")
            
            if self.use_tushare and self.stock_code in self.tushare_data:
                # 使用Tushare数据结合Alpha158
                return self._prepare_tushare_data()
            else:
                # 使用Qlib历史数据
                return self._prepare_qlib_data()
                
        except Exception as e:
            print(f"❌ 数据准备失败: {e}")
            return False
    
    def _prepare_tushare_data(self):
        """准备Tushare数据"""
        try:
            ts_code = self.tushare_data[self.stock_code]['ts_code']
            
            # 获取更长时间的历史数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')  # 获取约1.5年数据
            
            price_data = self.get_tushare_data(ts_code, start_date, end_date)
            
            if price_data.empty or len(price_data) < 100:
                print("⚠ Tushare数据不足，降级使用Qlib数据")
                return self._prepare_qlib_data()
            
            # 计算简化的技术指标 (Alpha158的子集)
            enhanced_data = self._calculate_simple_features(price_data)
            
            if enhanced_data.empty:
                return False
            
            # 分割训练和测试数据
            split_point = int(len(enhanced_data) * 0.8)
            train_data = enhanced_data.iloc[:split_point]
            test_data = enhanced_data.iloc[split_point:]
            
            print(f"✓ Tushare数据准备完成")
            print(f"训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")
            print(f"特征数量: {len(enhanced_data.columns) - 1}")
            
            # 保存到实例变量
            self.train_data = train_data
            self.test_data = test_data
            
            return True
            
        except Exception as e:
            print(f"❌ 准备Tushare数据失败: {e}")
            return False
    
    def _calculate_simple_features(self, price_data):
        """计算简化的技术指标"""
        try:
            df = price_data.copy()
            
            # 基础价格特征
            df['returns'] = df['$close'].pct_change()
            df['high_low_ratio'] = df['$high'] / df['$low']
            df['close_open_ratio'] = df['$close'] / df['$open']
            
            # 移动平均
            for window in [5, 10, 20]:
                df[f'ma_{window}'] = df['$close'].rolling(window).mean()
                df[f'price_ma_{window}_ratio'] = df['$close'] / df[f'ma_{window}']
            
            # 波动率
            for window in [5, 10, 20]:
                df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            
            # 成交量特征
            df['volume_ma_5'] = df['$volume'].rolling(5).mean()
            df['volume_ratio'] = df['$volume'] / df['volume_ma_5']
            
            # RSI简化版
            delta = df['$close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 标签: 未来1日收益率
            df['label'] = df['returns'].shift(-1)
            
            # 选择特征列，排除原始价格列和中间计算列
            exclude_cols = ['ma_5', 'ma_10', 'ma_20', 'volume_ma_5', 'ts_code']
            feature_cols = [col for col in df.columns 
                          if not col.startswith('$') and col not in exclude_cols]
            
            # 确保所有列都是数值型
            result_df = df[feature_cols].copy()
            for col in result_df.columns:
                if col != 'label':
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            
            result_df = result_df.dropna()
            
            print(f"✓ 计算技术指标完成，特征数: {len(result_df.columns)-1}")
            
            return result_df
            
        except Exception as e:
            print(f"❌ 计算技术指标失败: {e}")
            return pd.DataFrame()
    
    def _prepare_qlib_data(self):
        """准备Qlib数据"""
        try:
            train_end_date = '2020-06-30'
            
            # 使用Alpha158处理器
            self.handler = Alpha158(
                instruments=[self.stock_code],
                start_time=self.data_start_date,
                end_time=self.data_end_date,
                freq='day'
            )
            
            self.dataset = DatasetH(
                handler=self.handler,
                segments={
                    'train': (self.data_start_date, train_end_date),
                    'test': (train_end_date, self.data_end_date)
                }
            )
            
            # 获取数据
            train_data = self.dataset.prepare('train')
            test_data = self.dataset.prepare('test')
            
            print(f"✓ Qlib Alpha158数据准备完成")
            print(f"训练集: {train_data.shape}, 测试集: {test_data.shape}")
            
            # 保存到实例变量
            self.train_data = train_data
            self.test_data = test_data
            
            return True
            
        except Exception as e:
            print(f"❌ 准备Qlib数据失败: {e}")
            return False
    
    def train_models(self):
        """训练模型"""
        try:
            print("\n🤖 训练机器学习模型...")
            
            # 分离特征和标签
            if 'label' in self.train_data.columns:
                # Tushare数据格式
                label_col = 'label'
                feature_cols = [col for col in self.train_data.columns if col != 'label']
            else:
                # Qlib数据格式
                label_cols = [col for col in self.train_data.columns if 'LABEL' in str(col)]
                label_col = label_cols[0] if label_cols else self.train_data.columns[-1]
                feature_cols = [col for col in self.train_data.columns if 'LABEL' not in str(col)]
            
            X_train = self.train_data[feature_cols]
            y_train = self.train_data[label_col]
            X_test = self.test_data[feature_cols]
            y_test = self.test_data[label_col]
            
            # 处理NaN值
            X_train = X_train.fillna(X_train.mean()).fillna(0)
            X_test = X_test.fillna(X_train.mean()).fillna(0)
            y_train = y_train.fillna(y_train.median())
            y_test = y_test.fillna(y_train.median())
            
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
            data_source = "Tushare实时数据" if self.use_tushare else "Qlib历史数据"
            print(f"\n📅 分析基于: {data_source}")
            print(f"数据时间: {self.data_start_date} - {self.data_end_date}")
            
            print(f"\n⚠️ 风险提示:")
            print(f"• 预测基于历史数据模式，存在不确定性")
            print(f"• 投资决策请结合基本面和市场环境")
            print(f"• 投资有风险，入市需谨慎")
            
        except Exception as e:
            print(f"❌ 预测分析失败: {e}")
    
    def run_analysis(self):
        """运行分析"""
        print("🚀 增强股价预测器 - 集成Tushare数据源")
        print("=" * 70)
        
        if self.use_tushare:
            print("✓ 使用Tushare实时数据 + 机器学习模型")
        else:
            print("✓ 使用Qlib历史数据 + Alpha158因子")
        
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
        predictor = EnhancedStockPredictor()
        predictor.run_analysis()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n运行错误: {e}")

if __name__ == "__main__":
    main()