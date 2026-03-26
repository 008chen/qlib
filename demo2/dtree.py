"""
决策树零基础入门 - 极简水果分类
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt



# ========== 解决中文乱码 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ========== 第1步：创建超简单数据 ==========
# 数据：重量、长度
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

# 标签：0=苹果，1=香蕉
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

feature_names = ['重量', '长度(cm)']
class_names = ['苹果', '香蕉']

print("=== 我们的数据 ===")
df = pd.DataFrame(X, columns=feature_names)
df['水果'] = [class_names[i] for i in y]
print(df)


# ========== 第2步：训练决策树 ==========
# 创建模型（max_depth=2表示最多问2个问题）
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X, y)

print("=== 训练完成！ ===")
print(f"树的深度：{clf.get_depth()}")
print(f"叶子节点数：{clf.get_n_leaves()}")
print()

# ========== 第3步：看决策规则（纯文字版） ==========
print("=== 决策树规则（文字版） ===")
rules = export_text(clf, feature_names=feature_names)
print(rules)

# ========== 第4步：可视化树 ==========
plt.figure(figsize=(12, 8))
plot_tree(clf,
          feature_names=feature_names,
          class_names=class_names,
          filled=True,        # 填充颜色
          rounded=True,       # 圆角
          fontsize=12,
          impurity=False,     # 不显示基尼系数
          proportion=True)    # 显示比例
plt.title("水果分类决策树", fontsize=16)
plt.tight_layout()
plt.savefig('simple_tree.png', dpi=150, bbox_inches='tight')
plt.show()


