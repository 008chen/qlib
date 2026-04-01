from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL
import qlib
from typing import Optional, Union, List, Dict, Any
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
import matplotlib.pyplot as plt
import lightgbm as lgb


from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report 

import pandas as pd
from qlib.data.dataset.handler import DataHandlerLP

class CustomAlpha158(Alpha158):
    """
    最小化继承：只重写 get_feature_config 方法
    保留 Alpha158 的全部基础设施（数据加载、标签生成、预处理等）
    """
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=Alpha158.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        feature_whitelist: Optional[List[str]] = None,  # 白名单参数
        **kwargs
    ):
        """
        参数:
            feature_whitelist: 特征白名单列表，如 ['KMID', 'KLEN', 'ROC5', 'MA10']
                              如果为 None，则使用全部特征（与原始 Alpha158 行为一致）
        """
        self.feature_whitelist = feature_whitelist
      
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            process_type=process_type,
            filter_pipe=filter_pipe,
            inst_processors=inst_processors,
            **kwargs
        )
    
    # 分组定义特征，便于管理
    CUSTOM_FEATURES = {
        "momentum": {
            "ali_ibs": "100*($close - $low)/($high-$low  + 1e-12)",
            # "ali_atr": "($high-$low)/ATR($high, $low, $close, 14)",
        }
        # "volatility": {
        #     "MY_VOL_5": "Std($close, 5)",
        #     "MY_VOL_20": "Std($close, 20)",
        # },
        # "volume": {
        #     "MY_VP_RATIO": "$volume/Ref($volume, 1)",
        #     "MY_AMT_MA5": "Mean($amount, 5)",
        # }
    }

    
    def get_feature_config(self):
        # print(f"get_feature_config")
        # 获取原始特征配置
        conf = {
            "kbar": {}, # K线特征（蜡烛图特征）默认为空时生成全部
            "price": {
                "windows": [0],# 时间窗口，0 表示当前周期（当天），5 表示5天前
                "feature": ["OPEN", "HIGH", "LOW", "VWAP",'$close'],
            },
            "rolling": {},
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        
        
        # 展平嵌套字典
        my_names = []
        my_exprs = []
        
        for group_name, features in self.CUSTOM_FEATURES.items():
            for name, expr in features.items():
                my_names.append(name)
                my_exprs.append(expr)
        
        # # 如果没有设置白名单，返回全部特征
        if self.feature_whitelist is None or len(self.feature_whitelist) == 0:
            return my_exprs,my_names
        
        else:
            # 根据白名单过滤特征
            filtered_fields = []
            filtered_names = []
            
            for field, name in zip(fields, names):
                if name in self.feature_whitelist:
                    filtered_fields.append(field)
                    filtered_names.append(name)
            
            # 打印过滤信息
            print(f"原始特征数量: {len(names)}")
            print(f"白名单特征数量: {len(filtered_names)}")
            print(f"被过滤的特征: {set(names) - set(filtered_names)}")
            
            # base_exprs, base_names = super().get_feature_config()
            
        
            
            # print(f"{len(base_names)}+{len(my_names)}")
            return filtered_fields + my_exprs, filtered_names + my_names
        
    def get_label_config(self):
        # 自定义未来N日的收益率作为标签
        # Ref($close, -N) 表示未来第N天的收盘价
        return ["(Ref($high, -1)-$close) >($close - $low + 1e-12)"], ["LABEL0"]
    
    # def fetch(self, selector=None, level='datetime', col_set='feature', 
    #           data_key=DataHandlerLP.DK_I, **kwargs):
    #     """
    #     重写 fetch 方法，在返回数据时动态添加代码计算的特征
    #     """
    #     # 获取父类的数据
    #     df = super().fetch(selector=selector, level=level, col_set=col_set, 
    #                       data_key=data_key, **kwargs)
        
    #     print(f"selector:{selector}")
    #     print(f"level:{level}")
    #     print(f"================ {col_set} ================")
    #     print(f"DataFrame 形状: {df.shape}")
    #     print(f"列索引类型: {type(df.columns)}")
    #     print(f"所有列名: {df.columns.tolist()}")   
        
    #     # 如果是特征集，添加自定义代码特征
    #     if col_set == 'feature' or col_set == '__all':
    #         raw_df = D.features(self.instruments, fields= ["$open","$high","$low","$close"],start_time=selector[0],end_time=selector[1] )
    #         print(f"===============")
    #         print(raw_df)
    #         df = self.add_code_features(df)
    #     print(f"-------------------${col_set}--------------------------")
    #     print(df)
    #     return df
    
    # def add_code_features(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """使用 pandas 添加自定义特征"""
    #     # 创建副本避免修改原始数据
    #     df = df.copy()
        
    #     # 获取列名（根据实际数据格式调整）
    #     close_cols = [c for c in df.columns if 'close' in c.lower() or 'CLOSE' in c]
    #     if not close_cols:
    #         return df
        
    #     close_col = close_cols[0]
    #     close = df[close_col]
        
    #     # 按股票分组计算
    #     def calc_group(group):
    #         group = group.sort_index(level='datetime')
    #         c = group[close_col]
            
    #         # 代码计算的特征
    #         group['MA_RATIO_5_CODE'] = c.rolling(5).mean() / c
    #         group['RSI_CODE'] = self._calc_rsi(c)
    #         group['MOM_CODE'] = (c - c.shift(10)) / c.shift(10)
            
    #         return group
        
    #     # 应用分组计算
    #     df = df.groupby(level='instrument').apply(calc_group)
    #     df = df.dropna()
        
    #     return df
    
    # def _calc_rsi(self, series: pd.Series, period=14) -> pd.Series:
    #     """计算 RSI 的辅助方法"""
    #     delta = series.diff()
    #     gain = delta.clip(lower=0).rolling(period).mean()
    #     loss = (-delta.clip(upper=0)).rolling(period).mean()
    #     rs = gain / loss
    #     return 100 - (100 / (1 + rs))
    
  

def calc_cci(close, open_p, high, low, volume, period=20):
    """商品通道指数 (CCI)"""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    return cci 
   
feature_funcs = {
    'CCI': calc_cci,
}    
if __name__ == "__main__":
    # xx()
    
    import talib
    from qlib.data import D
    from qlib import init

    
    init(provider_uri="~/.qlib/qlib_data/cn_data" )
    


   
    from qlib.data.dataset import DatasetH
    
    
    whitelist = [
        # 'KMID', 'KLEN', 'KUP', 'KLOW',  # K线因子
        # 'ROC5', 'ROC10', 'ROC20',       # 收益率因子
        # 'MA5', 'MA10', 'MA20',          # 均线因子
        # 'STD5', 'STD10', 'STD20',       # 标准差因子
        # 'RSI5', 'RSI10', 'RSI20',       # RSI因子
    ]
    
    # 实例化：所有参数与 Alpha158 兼容
    handler = CustomAlpha158(
        start_time = "2020-01-01",
        end_time = "2026-03-20",
        fit_start_time="2020-01-01",
        fit_end_time="2024-01-01",
        instruments=["SZ002285","SH601577"],
        infer_processors=[],  # 可自定义预处理器
        learn_processors=[],  # 可自定义标签处理器
        # feature_whitelist = whitelist,  # 传入白名单
        # feature_registry=feature_funcs
    )
    
    # 获取数据
    dataset = DatasetH(handler=handler, segments={
        "train": ("2020-01-01", "2024-01-01"),
        "valid": ("2024-01-01", "2025-01-01"),
        "test": ("2026-01-01", "2026-01-12"),
    })
    
    # df_train = dataset.prepare("train")
   
    # print(f"特征数量: {len(df_train.columns)}")  
    # print(f"新增特征: {[c for c in df_train.columns if c.startswith('ali_')]}")
    
    
    print("创建LightGBM模型...")
    model = LGBModel(
        loss='binary',                         #损失函数：均方误差，适用于回归任务
                                            # 其他选项：'mae'（绝对误差）、'huber'（鲁棒回归）
        scale_pos_weight=50,                                      
        colsample_bytree=0.8879,            #特征采样比例 每棵树随机使用88.79%的特征
                                            # 作用：增加随机性，防止过拟合，类似随机森林
        learning_rate=0.2,                  #学习率  控制每棵树的贡献权重
                                            # 较大的学习率（0.2）训练快但可能欠拟合，常用0.01
                                            
        subsample=0.8789,                   #样本采样比例  每棵树随机使用87.89%的样本
                                            # 作用：Bagging采样，减少方差，防止过拟合
                                            
        n_estimators=100,                   #迭代次数）：总共构建100棵树
                                            # 与学习率配合：小学习率需要更多树
                                            
                                            
        max_depth=1,                        #树的最大深度：限制8层
                                            # 作用：控制模型复杂度，防止过拟合，常用3-12
                                            
        num_leaves=2,                     #叶子节点数  LightGBM特有的参数
                                            # 控制模型复杂度，通常设置为(2^max_depth)附近
                                            
        min_child_samples=20,               #最小样本数
        
        # reg_alpha=0.0,            # L1正则化系数（代码中未设置，默认0）
        # reg_lambda=0.0,             # L2正则化系数（代码中未设置，默认0）
        # random_state=42,            # 随机种子（代码中未设置）
        verbose=-1                 # 训练过程输出控制：-1表示不输出训练日志
    )
    
    # feature_names = handler.get_feature_config()[1]
    # print(feature_names)

    # ==================== 第六步：训练模型 ====================
    # 训练模型
    print("训练模型...")
    model.fit(dataset)  # Qlib的fit方法直接使用DatasetH对象
                        # 内部会自动处理训练集和验证集，支持早停（early stopping）
                        
                        
                        
    print("准备预测..") 
    pred = model.predict(dataset, segment='test')# 对测试集进行预测
                                                 # 返回预测值序列（未来收益率预测）

   
                                                 
    pred_labels = (pred >= 0.5).astype(int)
                                                 
    test_data = dataset.prepare('test')
    
    X_test = test_data.drop('LABEL0', axis=1)
    y_test = test_data['LABEL0']
    
    print(y_test)
    accuracy = accuracy_score(y_test, pred_labels)
    print(f"测试集精度: {accuracy:.4f}")
    
    print(classification_report(y_test, pred_labels))
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # 显示混淆矩阵
    print(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
                
    
    booster = model.model  
    
    
    
    import graphviz
    import lightgbm as lgb

    # 假设 booster 是你从上一步获取的 lightgbm.Booster 对象

    # 1. 指定要绘制的树的索引，0 代表第一棵树
    tree_index = 0

    # 2. 创建 Graphviz 图形对象
    #    show_info 参数是信息全面的关键，可以添加多种信息到节点上
    graph = lgb.create_tree_digraph(
        booster, 
        tree_index=tree_index,
        show_info=[
        'split_gain',      # 分裂增益
        'internal_count',  # 内部节点样本数
        'internal_value',  # 节点预测值
        'leaf_count',      # 叶子节点样本数
        'leaf_weight',     # 叶子节点总权重
        'internal_weight', # 内部节点总权重
        'data_percentage'  # 数据百分比
        ]
    )

    # 3. 渲染并查看图形
    #    这会生成一个 .gv 文件，并尝试用系统默认的 PDF 阅读器打开
    graph.render('./alinqmt/my_qlib_lgb_tree', format='png', cleanup=True)
    print("决策树图已保存为 my_qlib_lgb_tree.png")
        


   