#!/usr/bin/env python3
"""
LightGBM基础演示
基于第3章监督学习模型内容
"""

import qlib
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
import numpy as np

if __name__ == '__main__':
    
    # ==================== 第一步：初始化Qlib环境 ====================
    # 初始化Qlib
    print("初始化Qlib...")
    qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")
    
    
    # ==================== 第二步：配置数据处理器 ====================
    # Alpha158是Qlib内置的经典特征集，包含158个基于量价的技术指标
    # 准备数据处理器
    handler = Alpha158(
        instruments='csi300',
        start_time='2020-01-01',
        end_time='2020-12-31',
        freq='day'
    )
    # Alpha158特征包括：收盘价、成交量、换手率、MACD、RSI、布林带等技术指标
    
    # 创建数据集
    print("创建数据集...")
    dataset = DatasetH(
        handler=handler,   # 使用上面定义的数据处理器
        segments={     # 定义三个数据时段（避免未来数据泄露）
            'train': ('2020-01-01', '2020-06-30'),  # 训练集：上半年数据
            'valid': ('2020-07-01', '2020-09-30'),  # 验证集：第三季度，用于调参
            'test': ('2020-10-01', '2020-12-31')    # 测试集：第四季度，用于最终评估
        }
    )
    # 时间序列数据必须按时间顺序划分，不能用随机分割，防止信息泄露
    
    
    # ==================== 第四步：准备数据 ====================
    # 准备训练数据
    print("准备数据...")
    train_data = dataset.prepare('train') # 加载训练集数据（DataFrame格式）
    valid_data = dataset.prepare('valid')
    test_data = dataset.prepare('test')
    
    # 分离特征和标签
    # 'LABEL0'是Qlib默认的预测目标，通常是未来收益率（如次日收益率）
    X_train = train_data.drop('LABEL0', axis=1) # 删除标签列，保留所有特征
    y_train = train_data['LABEL0']              # 提取标签：预测目标
    X_valid = valid_data.drop('LABEL0', axis=1)
    y_valid = valid_data['LABEL0']
    X_test = test_data.drop('LABEL0', axis=1)
    y_test = test_data['LABEL0']
    
    
    # 打印数据维度信息
    print(f"训练集大小: {X_train.shape}") # 输出：(样本数, 特征数)
    print(f"验证集大小: {X_valid.shape}")
    print(f"测试集大小: {X_test.shape}")


    # ==================== 第五步：配置LightGBM模型参数 ====================
    # 创建LightGBM模型
    print("创建LightGBM模型...")
    model = LGBModel(
        loss='mse',                         #损失函数：均方误差，适用于回归任务
                                            # 其他选项：'mae'（绝对误差）、'huber'（鲁棒回归）
                                            
        colsample_bytree=0.8879,            #特征采样比例 每棵树随机使用88.79%的特征
                                            # 作用：增加随机性，防止过拟合，类似随机森林
        learning_rate=0.2,                  #学习率  控制每棵树的贡献权重
                                            # 较大的学习率（0.2）训练快但可能欠拟合，常用0.01
                                            
        subsample=0.8789,                   #样本采样比例  每棵树随机使用87.89%的样本
                                            # 作用：Bagging采样，减少方差，防止过拟合
                                            
        n_estimators=100,                   #迭代次数）：总共构建100棵树
                                            # 与学习率配合：小学习率需要更多树
                                            
                                            
        max_depth=8,                        #树的最大深度：限制8层
                                            # 作用：控制模型复杂度，防止过拟合，常用3-12
                                            
        num_leaves=210,                     #叶子节点数  LightGBM特有的参数
                                            # 控制模型复杂度，通常设置为(2^max_depth)附近
                                            
        min_child_samples=20,               #最小样本数
        
        # reg_alpha=0.0,            # L1正则化系数（代码中未设置，默认0）
        # reg_lambda=0.0,             # L2正则化系数（代码中未设置，默认0）
        # random_state=42,            # 随机种子（代码中未设置）
        verbose=-1                 # 训练过程输出控制：-1表示不输出训练日志
    )
    
    
    # ==================== 第六步：训练模型 ====================
    # 训练模型
    print("训练模型...")
    model.fit(dataset)  # Qlib的fit方法直接使用DatasetH对象
                        # 内部会自动处理训练集和验证集，支持早停（early stopping）
    
    
    # ==================== 第七步：模型预测 ====================
    # 预测
    print("进行预测...")
    pred = model.predict(dataset, segment='test')# 对测试集进行预测
                                                 # 返回预测值序列（未来收益率预测）
                                                 
    print("LightGBM预测结果样例：")
    print(pred[:10])  # 显示前10个预测结果
    
    # ==================== 第八步：性能评估 ====================
    # 评估指标1：均方误差（MSE）- 衡量预测值与真实值的偏差程度
    mse = np.mean((pred.values.flatten() - y_test.values) ** 2)
    print(f"测试集MSE: {mse:.6f}")
    # MSE越小越好，但对异常值敏感
    
    
    
    # 评估指标2：信息系数（IC）- 量化投资核心指标
    correlation = np.corrcoef(pred.values.flatten(), y_test.values)[0, 1]
    print(f"IC (信息系数): {correlation:.4f}")
    # IC衡量预测值与真实收益率的相关性
    # |IC| > 0.03 通常认为有效，|IC| > 0.05 较好，|IC| > 0.1 优秀
    # 正IC表示预测方向正确，负IC表示反向关系
    
    print("LightGBM演示完成！")