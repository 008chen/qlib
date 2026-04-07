import qlib
import pandas as pd
import talib
import numpy as np
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
from qlib.data import D

import alin_datahandler 

from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report 
import matplotlib.pyplot as plt


stock_list=["SZ002285","SH601577"]

start_time = '2026-01-01'
end_time='2026-02-15'

segments = {
    "train": (start_time, end_time),
    # "valid": ("2023-01-01", "2023-06-30"),
    # "test": ("2023-07-01", end_time)
}
    
if __name__ == "__main__":
    # 1. 初始化 Qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')

    # 2. 获取 Alpha158 数据
    alpha158_handler = alin_datahandler.CustomAlpha158(
        instruments=stock_list,
        start_time=start_time,
        end_time=end_time,
        freq="day",
        infer_processors=[],  # 可自定义预处理器
        learn_processors=[],  # 可自定义标签处理器
    )
    

    # 获取 Alpha158 的原始数据（包含特征和标签）
    alpha158_data = alpha158_handler.fetch(data_key=DataHandlerLP.DK_L)

    # 3. 计算 TA-Lib 指标
    # instruments = D.instruments(market='csi300')
    raw_data = D.features(
        instruments=stock_list, 
        fields=['$close', '$high', '$low', '$open', '$volume'], 
        start_time=start_time, 
        end_time=end_time
    )
 

    def calc_talib_features(group):
        group = group.dropna()
        close = group['$close'].values.astype(np.float64)
        high = group['$high'].values.astype(np.float64)
        low = group['$low'].values.astype(np.float64)
        
        features = pd.DataFrame(index=group.index)
   
        # features['ATR14'] = talib.ATR(high, low, close, 14)

        return features
    
    # print(raw_data)
    talib_features = raw_data.groupby(level='instrument').apply(calc_talib_features)
    talib_features = talib_features.droplevel(0)
    talib_features = talib_features.reset_index().set_index(['datetime', 'instrument']).sort_index()
    
    # talib_features = talib_features.dropna()
    # print(talib_features)
    # print(raw_data)
    
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # print(talib_features)
    # print("alpha158_data")
    # print(alpha158_data)

    # 5. 使用 DataHandlerLP.from_df() 创建 Handler
    # 需要将数据转换为正确的 MultiIndex 格式
    # 假设 alpha158_data 已经有 (feature, col) 和 (label, col) 的列结构

    # 如果 talib_features 没有 MultiIndex，需要转换


    
    
    # 重新合并
    combined_data = alpha158_data.join(talib_features, how='inner')

    
    new_columns = []
    label_cols = ['LABEL0']
    for col in combined_data.columns.get_level_values(0):
        if col not in label_cols:
            new_columns.append(('feature', col))
        else:
            new_columns.append(('label', col))

    combined_data.columns = pd.MultiIndex.from_tuples(new_columns)
    
    print(combined_data)
    

    # 创建 Handler
    handler = DataHandlerLP.from_df(combined_data)
    # handler.feature_names = ["ali_ibs"]

    # 6. 创建 DatasetH 并训练
    dataset = DatasetH(handler=handler, segments=segments)

 
    model = LGBModel(
        loss='binary',                         #损失函数：均方误差，适用于回归任务
                                            # 其他选项：'mae'（绝对误差）、'huber'（鲁棒回归）
        # metric= "binary_logloss",
        # scale_pos_weight = 10.0,          # 关键参数：增大此值会让模型更保守地预测正类
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
                                            
        # num_leaves=2,                     #叶子节点数  LightGBM特有的参数
                                            # 控制模型复杂度，通常设置为(2^max_depth)附近
                                            
        min_child_samples=20,               #最小样本数
        
        # reg_alpha=0.0,            # L1正则化系数（代码中未设置，默认0）
        # reg_lambda=0.0,             # L2正则化系数（代码中未设置，默认0）
        # random_state=42,            # 随机种子（代码中未设置）
        verbose=2                 # 训练过程输出控制：-1表示不输出训练日志
    )
    
    # feature_names = handler.get_feature_config()[1]
    # print(feature_names)

    # ==================== 第六步：训练模型 ====================
    # 训练模型
    print("训练模型...")
    model.fit(dataset)  # Qlib的fit方法直接使用DatasetH对象
                        # 内部会自动处理训练集和验证集，支持早停（early stopping）
                        
                        
                        
    print("准备预测..") 
    pred = model.predict(dataset, segment='train')# 对测试集进行预测
                                                 # 返回预测值序列（未来收益率预测）

   
                                                 
    pred_labels = (pred >= 0.5).astype(int)
    print(pred_labels)
                                                 
    test_data = dataset.prepare('train')
    
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
    
    # model_dict = booster.dump_model()
    # model_dict['feature_names'] = ["xxxx"]
    
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
# ┌─────────────────────────┐
# │ Column_0 ≤ 58.571       │  ← 分裂条件：特征"Column_0"的值是否≤58.571
# │ 14.816 gain             │  ← 分裂增益（信息增益）
# │ 0.134 value             │  ← 该节点的预测值（raw score，未sigmoid）
# │ 14.933 weight           │  ← 该节点的样本权重之和
# │ count: 60               │  ← 该节点包含的样本数
# │ 100.00% of data         │  ← 占父节点数据的比例（根节点就是100%）
# └─────────────────────────┘