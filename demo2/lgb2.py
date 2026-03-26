import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import  plot_tree
import pandas as pd
# 构造数据
# X = np.array([[1], [2], [3], [4]])  # 特征
# y = np.array([0, 0, 1, 1])          # 标签
# ========== 解决中文乱码 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

X = np.array([
    [150, 7],    # 苹果
    [170, 7.5],  # 苹果
    [140, 6.8],  # 苹果
    [180, 8],    # 苹果
    [120, 9],   # 香蕉
    [130, 16],   # 香蕉
    [110, 14],   # 香蕉
    [140, 17],   # 香蕉
])



# 定义列名 (例如：重量克数, 甜度)
columns = ['weight', 'length']

# 定义索引标签 (可选，用于标记是苹果还是香蕉)
index = ['Apple_1', 'Apple_2', 'Apple_3', 'Apple_4', 
         'Banana_1', 'Banana_2', 'Banana_3', 'Banana_4']

# 转换为 DataFrame
df_x = pd.DataFrame(X, columns=columns, index=index)


# 标签：0=苹果，1=香蕉
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])



# 创建 Dataset
train_data = lgb.Dataset(df_x, label=y)




# 设置参数（极简）
params = {
    'objective': 'binary',#我现在要解决的是一个二分类问题,
    'verbose': 2,  
    'min_data_in_leaf': 1,  # 允许极小叶子（默认可能为20，需调小）
    'min_child_samples': 1  # 兼容旧版本参数
}


callbacks = [
    # 1. 打印日志：每 1 轮打印一次
    # lgb.early_stopping(stopping_rounds=5, verbose=True), 
    lgb.log_evaluation(period=1) # period=1 等同于 verbose_eval=1
]

# 训练模型
model = lgb.train(params, train_data, num_boost_round=1, callbacks=callbacks)

# 查看分裂信息
print("根节点分裂特征:", model.feature_importance())
print("模型结构:")
print(model.trees_to_dataframe())
print(f"总共几颗树:{model.num_trees()}")

# graph = lgb.create_tree_digraph(model, tree_index=0)
# graph.render('lightgbm_tree', format='dot', cleanup=True)


for i in range(model.num_trees()):
    graph = lgb.create_tree_digraph(model, tree_index=i)
    graph.render(f'tree_{i}', format='png', cleanup=True)
    print(f"Tree {i} rendered.")



