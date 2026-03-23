import qlib
from qlib.contrib.model.catboost_model import CatBoostModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
import numpy as np

if __name__ == '__main__':
    # 初始化Qlib
    qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")
    
    # 准备数据处理器
    handler = Alpha158(
        instruments='csi300',
        start_time='2020-01-01',
        end_time='2020-12-31',
        freq='day'
    )
    
    # 创建数据集
    dataset = DatasetH(
        handler=handler,
        segments={
            'train': ('2020-01-01', '2020-06-30'),
            'valid': ('2020-07-01', '2020-09-30'),
            'test': ('2020-10-01', '2020-12-31')
        }
    )
    
    # 准备训练数据
    train_data = dataset.prepare('train')
    valid_data = dataset.prepare('valid')
    test_data = dataset.prepare('test')
    
    # 分离特征和标签
    X_train = train_data.drop('LABEL0', axis=1)
    y_train = train_data['LABEL0']
    X_valid = valid_data.drop('LABEL0', axis=1)
    y_valid = valid_data['LABEL0']
    X_test = test_data.drop('LABEL0', axis=1)
    y_test = test_data['LABEL0']

    # 创建CatBoost模型
    model = CatBoostModel(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        random_seed=42
    )
    
    # 训练模型（CatBoost需要DatasetH对象）
    model.fit(dataset)
    
    # 预测
    pred = model.predict(dataset, segment='test')
    print("CatBoost预测结果样例：")
    print(pred.head(10))
    
    # 计算简单的性能指标
    mse = np.mean((pred.values.flatten() - y_test.values) ** 2)
    print(f"测试集MSE: {mse:.6f}")
    
    # 计算IC（信息系数）
    correlation = np.corrcoef(pred.values.flatten(), y_test.values)[0, 1]
    print(f"IC (信息系数): {correlation:.4f}")