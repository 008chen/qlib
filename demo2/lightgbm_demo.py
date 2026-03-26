"""
Qlib + LightGBM 极简可运行示例（完全绕过配置问题）
- 1只股票，10个交易日
- 无复杂 DataLoader 配置
- 直接手动准备数据
"""

import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
import lightgbm as lgb
# import qlib
# from qlib.data.dataset import DatasetH
# from qlib.contrib.model.gbdt import LGBModel

# 初始化 Qlib
# qlib.init()

# ============================================
# 第1步：手动构造数据（完全透明）
# ============================================

dates = pd.date_range(start='2024-01-01', periods=4, freq='B')
stock_code = 'SH600000'

# 构造数据
records = []
for i, date in enumerate(dates):
    # close = 10.0 + i * 0.5  # 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5
    # volume = 1000 + i * 100  # 1000, 1100, 1200...
    close = i
    
    records.append({
        'date': date,
        'instrument': stock_code,
        'close': close,
        # 'volume': volume,
    })

df = pd.DataFrame(records)
df.set_index(['date', 'instrument'], inplace=True)

# 手动计算标签：未来1日收益率
# df['label'] = df['close'].shift(-1) / df['close'] - 1
# df['label'] = df['label'].fillna(0)
df['label']=[0,0,1,1]
print(df)

# print("=" * 70)
# print("原始数据（10行，可逐行手工验证）：")
# print("=" * 70)
# print(df.to_string())
# print(f"\n验证计算：")
# print(f"  第1天 label = 10.5/10.0 - 1 = {10.5/10.0 - 1:.4f} (5.00%)")
# print(f"  第2天 label = 11.0/10.5 - 1 = {11.0/10.5 - 1:.4f} (4.76%)")
# print(f"  第3天 label = 11.5/11.0 - 1 = {11.5/11.0 - 1:.4f} (4.55%)")

# ============================================
# 第2步：手动准备 DatasetH（绕过复杂 Handler）
# ============================================

# 直接构造 DatasetH 需要的内部数据格式
# 特征矩阵 X 和标签 y

# 划分训练/测试：前7天训练，后3天测试
train_df = df.iloc[:7].copy()
test_df = df.iloc[7:].copy()

# 构造特征和标签
# feature_cols = ['close', 'volume']
feature_cols = ['close', ]
label_col = 'label'

# 创建符合 Qlib 格式的数据对象
class SimpleDataset:
    """
    极简数据集对象，兼容 Qlib 的 LGBModel
    """
    def __init__(self, train_df, test_df, feature_cols, label_col):
        self.train_df = train_df
        self.test_df = test_df
        self.feature_cols = feature_cols
        self.label_col = label_col
        
        # 构造 segments 属性（LGBModel 需要）
        self.segments = {
            "train": ("train", train_df),
            "test": ("test", test_df)
        }
    
    def prepare(self, segment, col_set=None):
        """
        准备数据，返回 (X, y) 或 DataFrame
        """
        if segment == "train":
            data = self.train_df
        elif segment == "test":
            data = self.test_df
        else:
            raise ValueError(f"Unknown segment: {segment}")
        
        if col_set == "feature":
            return data[self.feature_cols]
        elif col_set == "label":
            return data[[self.label_col]]
        else:
            # 返回特征和标签
            return data[self.feature_cols + [self.label_col]]

# 创建数据集
dataset = SimpleDataset(train_df, test_df, feature_cols, label_col)

# 查看数据
print("\n" + "=" * 70)
print("训练集特征（前7天）：")
print("=" * 70)
print(dataset.prepare("train", col_set="feature").to_string())

print("\n训练集标签：")
print(dataset.prepare("train", col_set="label").to_string())

# ============================================
# 第3步：手动提取数据训练 LightGBM
# ============================================

print("\n" + "=" * 70)
print("使用 LightGBM 训练（绕过 Qlib LGBModel 的复杂逻辑）：")
print("=" * 70)



# 准备 numpy 数据
X_train = dataset.prepare("train", col_set="feature").values
y_train = dataset.prepare("train", col_set="label").values.ravel()
X_test = dataset.prepare("test", col_set="feature").values
y_test = dataset.prepare("test", col_set="label").values.ravel()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")

# 训练模型
model = lgb.LGBMRegressor(
    n_estimators=3,      # 仅3棵树
    max_depth=2,         # 最大深度2
    learning_rate=0.1,
    num_leaves=4,
    verbose=-1,
    random_state=42
)

model.fit(X_train, y_train)

model_dump = model.booster_.dump_model()
trees = model_dump.get('tree_info', [])

if not trees:
    print("❌ 模型中没有找到任何树信息。")
else:
    print(f"✅ 模型共包含 {len(trees)} 棵树。")
    # 3. 检查第一棵树 (index 0)
    tree_0 = trees[0]
    
    # 判断是否有分裂：如果有 'split_feature' 键且列表不为空，说明有分裂
    # 对于单节点树，通常只有 'internal_value' 而没有 'split_feature' 或该列表为空
    has_split = 'split_feature' in tree_0 and len(tree_0['split_feature']) > 0
    
    if has_split:
        print("✅ 第一棵树包含分裂节点，可以尝试可视化。")
        # 这里可以再次调用 plot_tree
    else:
        print("⚠️ 第一棵树没有分裂节点 (只有根节点)。")
        print(f"   根节点预测值: {tree_0.get('internal_value')}")
        
    # 4. (可选) 寻找第一棵有分裂的树的索引
    valid_tree_index = -1
    for i, tree in enumerate(trees):
        if 'split_feature' in tree and len(tree['split_feature']) > 0:
            valid_tree_index = i
            break
    
    if valid_tree_index != -1:
        print(f"💡 建议尝试可视化第 {valid_tree_index} 棵树 (tree_index={valid_tree_index})")
    else:
        print("❌ 所有树都没有分裂节点，请检查训练数据或参数。")
# plt.figure(figsize=(20, 10))
# lgb.plot_tree(model, tree_index=0, show_info=['split_gain', 'internal_value'])
# plt.title("LGBM Tree (Matplotlib)")
# plt.show()

# # ============================================
# # 第4步：预测与验证
# # ============================================

# train_pred = model.predict(X_train)
# test_pred = model.predict(X_test)

# print("\n" + "=" * 70)
# print("预测结果（可手工验证）：")
# print("=" * 70)

# print("\n【训练集】前7天：")
# train_features = dataset.prepare("train", col_set="feature")
# for i in range(len(X_train)):
#     date = train_features.index[i][0].strftime('%m-%d')
#     close = X_train[i][0]
#     actual = y_train[i]
#     pred = train_pred[i]
#     print(f"  {date}: close={close:.1f}, 实际={actual:+.4f}, 预测={pred:+.4f}, 误差={abs(actual-pred):.4f}")

# print("\n【测试集】后3天：")
# test_features = dataset.prepare("test", col_set="feature")
# for i in range(len(X_test)):
#     date = test_features.index[i][0].strftime('%m-%d')
#     close = X_test[i][0]
#     actual = y_test[i]
#     pred = test_pred[i]
#     print(f"  {date}: close={close:.1f}, 实际={actual:+.4f}, 预测={pred:+.4f}, 误差={abs(actual-pred):.4f}")

# # 特征重要性
# print("\n" + "=" * 70)
# print("特征重要性：")
# print("=" * 70)
# importance = model.feature_importances_
# print(f"  close:   {importance[0]}")
# print(f"  volume:  {importance[1]}")

# print("\n✅ 完成！所有数据均可手工验证，无黑盒逻辑。")