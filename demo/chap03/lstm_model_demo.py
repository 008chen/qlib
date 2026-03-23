#!/usr/bin/env python3
"""
LSTM模型示例
基于第3章深度学习模型内容
"""

import qlib
import torch
import torch.nn as nn
import numpy as np
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.pytorch_nn import DNNModelPytorch
from qlib.data.dataset import DatasetH

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 确保输入是3维的 (batch_size, seq_len, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加时间维度
        
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # Dropout和全连接层
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

def main():
    print("准备数据...")
    
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
    
    print("创建LSTM模型...")
    
    # 获取特征数量
    train_data = dataset.prepare('train')
    feature_dim = train_data['feature'].shape[1]
    
    # 创建LSTM模型
    model = LSTMModel(
        input_size=feature_dim,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    )
    
    # 包装为Qlib模型
    qlib_model = DNNModelPytorch(
        model=model,
        optimizer='adam',
        loss='mse',
        lr=0.001,
        max_epochs=50,
        batch_size=256,
        early_stop=10,
        GPU=0 if torch.cuda.is_available() else None
    )
    
    print("训练模型...")
    qlib_model.fit(dataset)
    
    print("模型预测...")
    predictions = qlib_model.predict(dataset, segment='test')
    
    print("预测结果样例：")
    print(predictions.head(10))
    
    print("模型评估...")
    # 获取测试数据的真实标签
    test_data = dataset.prepare('test')
    y_true = test_data['label']['LABEL0']
    
    # 计算MSE
    mse = np.mean((predictions.values.flatten() - y_true.values.flatten()) ** 2)
    print(f"测试集MSE: {mse:.6f}")
    
    # 计算相关系数
    correlation = np.corrcoef(predictions.values.flatten(), y_true.values.flatten())[0, 1]
    print(f"预测相关系数: {correlation:.6f}")

if __name__ == "__main__":
    main()