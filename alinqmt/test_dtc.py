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
from sklearn.tree import DecisionTreeClassifier, plot_tree
import graphviz
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

import features

stock_list=["SZ002285"]
# stock_list="all"
start_time = '2020-01-01'
end_time = '2026-01-01'
# end_time='2026-03-20'

segments = {
    "train": (start_time, end_time),
    "valid": ("2023-01-01", "2023-06-30"),
    "test": ("2023-07-01", end_time)
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

    instruments = D.instruments(market=stock_list)

    # 3. 计算 TA-Lib 指标
    # '$volume','$factor','Ref(IdxMax($high, 20), -20)','Ref(IdxMin($low, 20), -20)'
    raw_data = D.features(
        instruments=instruments, 
        fields=['$close/$factor','$high/$factor', '$low/$factor', '$open/$factor' ], 
        start_time=start_time, 
        end_time=end_time
    )
    raw_data.rename(columns={'$close/$factor': 'close', '$high/$factor': 'high','$low/$factor': 'low','$open/$factor': 'open'}, inplace=True)
   

    talib_features = raw_data.groupby(level='instrument').apply(features.calc_talib_features)
    talib_features = talib_features.droplevel(0)
    talib_features = talib_features.reset_index().set_index(['datetime', 'instrument']).sort_index()

    
    # 重新合并
    combined_data = alpha158_data.join(talib_features, how='inner')
    

    combined_data.drop(['open','high','low','close','LABEL0','ali_inside_bar'], axis=1, inplace=True)

    new_columns = []
   
    for col in combined_data.columns.get_level_values(0):
        if not col.startswith("LABEL"):
            new_columns.append(('feature', col))
        else:
            new_columns.append(('label', col))

    combined_data.columns = pd.MultiIndex.from_tuples(new_columns)
    
    with pd.option_context('display.max_rows', None, 
                       'display.max_columns', None,
                       'display.width', None):
        print(combined_data)
        print(combined_data.head(1))
    # print(combined_data.describe())

    # 创建 Handler
    handler = DataHandlerLP.from_df(combined_data)

    # 6. 创建 DatasetH 并训练
    dataset = DatasetH(handler=handler, segments=segments)

 
    # ==================== 关键修改：使用 sklearn 决策树 ====================
    
    # 准备数据（转换为 sklearn 格式）
    print("\n准备训练数据...")
    train_data = dataset.prepare("train")
    
    use_label= 'LABEL_BO'
    # 分离特征和标签
    X_train = train_data.drop(use_label, axis=1)
    y_train = train_data[use_label]
    
    # 获取特征名列表（关键！用于可视化显示）
    feature_names = X_train.columns.tolist()
    print(f"特征数量: {len(feature_names)}")
    print(f"特征名: {feature_names[:]}...")  # 显示前5个
    
    # 转换为 numpy 数组（sklearn 需要）
    X_train_np = X_train.values
    y_train_np = y_train.values
    
    
    test_data = dataset.prepare("test")
    # todo
    test_data = train_data
    X_test = test_data.drop(use_label, axis=1)
    y_test = test_data[use_label]
    X_test_np = X_test.values
    y_test_np = y_test.values
    
    # 6. 创建并训练普通决策树
    print("\n训练决策树模型...")
    model = DecisionTreeClassifier(
        criterion='gini',        # 分裂标准：'gini' 或 'entropy'
        max_depth=2,             # 树的最大深度（防止过拟合）
        min_samples_split=2,    # 内部节点再划分所需最小样本数
        min_samples_leaf=2,     # 叶子节点最小样本数
        random_state=42,         # 随机种子（保证可复现）
        # class_weight={0: 1, 1: 0.3},
        # class_weight='balanced'  # 处理类别不平衡
    )

    # ==================== 第六步：训练模型 ====================
    # 训练模型
    print("训练模型...")
    model.fit(X_train_np, y_train_np)
                        
                        
                        
    print("准备预测..") 
    pred = model.predict(X_test_np)
    pred_proba = model.predict_proba(X_test_np)[:, 1]  # 正类概率
    
    print(f"预测结果: {pred}")
    print(f"正类概率: {pred_proba}")
    
    print(f"类别比例分布: {test_data[use_label].value_counts(normalize=True)}")
                                                 
 # 评估
    accuracy = accuracy_score(y_test_np, pred)
    print(f"\n测试集精度: {accuracy:.4f}")
    print("\n分类报告：")
    print(classification_report(y_test_np, pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test_np, pred)
    print(f"\n混淆矩阵:\n{cm}")
    
    # cm_ratio = cm / cm.sum()
    # print(f"\n混淆矩阵（整体比例）:\n{cm_ratio.round(4)}")
    
    # cm_row_ratio = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(f"\n混淆矩阵（行比例 - Recall）:\n{cm_row_ratio.round(4)}")
    
    # cm_col_ratio = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    # print(f"\n混淆矩阵（列比例 - Precision）:\n{cm_col_ratio.round(4)}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # ==================== 决策树可视化（多种方式）====================
    
    # 方式一：matplotlib 直接绘制（简单，但美观度一般）
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,      # 显示真实特征名！
        class_names=['nan','Up','Down'],         # 类别名称
        filled=True,                        # 填充颜色
        rounded=True,                       # 圆角节点
        fontsize=10,
        precision=4,
        max_depth=3                         # 限制显示深度
    )
    plt.title('Decision Tree Visualization (Matplotlib)')
    plt.savefig('./alinqmt/decision_tree_mpl.png', dpi=300, bbox_inches='tight')
    print("\n决策树图已保存: decision_tree_mpl.png")
    plt.show()

    # 方式二：使用 graphviz（更美观，支持 PDF/SVG）
    print("\n生成 Graphviz 可视化...")
    dot_data = tree.export_graphviz(
        model,                          # 决策树模型
        out_file=None,                  # 不输出文件，直接返回字符串
        feature_names=feature_names,    # 特征名（已设置）
        class_names=['nan','Up','Down'],     # 类别名
        filled=True,                    # ✅ 按类别填充颜色
        rounded=True,                   # ✅ 圆角节点
        special_characters=True,        # ⭐ 新增：支持特殊字符
        
        # 信息控制参数
        impurity=True,                  # 显示 gini/entropy（默认）
        node_ids=True,                  # ⭐ 新增：显示节点编号如 "node #0"
        proportion=True,                # ⭐ 新增：显示样本比例（如 50.0%）
        
        # 布局参数
        rotate=False,                   # ⭐ 可改为 True 横向显示
        leaves_parallel=False,          # ⭐ 可改为 True 对齐叶子节点
        
        precision=4,                    # 小数位数（默认3）
    )
    
    # 渲染为 PNG
    graph = graphviz.Source(dot_data)
    graph.render('./alinqmt/decision_tree_graphviz', format='png', cleanup=True)
    print("决策树图已保存: decision_tree_graphviz.png")
    
    # 也可以保存为 PDF（矢量图，更清晰）
    graph.render('./alinqmt/decision_tree_graphviz', format='pdf', cleanup=True)
    print("决策树图已保存: decision_tree_graphviz.pdf")

    # 方式三：文本形式展示树结构（快速查看）
    print("\n" + "="*50)
    print("决策树文本结构：")
    print("="*50)
    tree_rules = tree.export_text(model, feature_names=feature_names, max_depth=3)
    print(tree_rules)
    
    # 保存文本规则
    with open('./alinqmt/decision_tree_rules.txt', 'w') as f:
        f.write(tree_rules)
    print("决策树规则已保存: decision_tree_rules.txt")

    # 8. 特征重要性分析
    print("\n" + "="*50)
    print("特征重要性 Top 10：")
    print("="*50)
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(10)
    print(top_features)
    
    # 绘制特征重要性
    plt.figure(figsize=(10, 6))
    top_features.plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('./alinqmt/feature_importance.png', dpi=300)
    print("特征重要性图已保存: feature_importance.png")
    plt.show()
    
    