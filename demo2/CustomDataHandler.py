import pandas as pd
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset.processor import Processor
from typing import List, Optional, Union, Dict, Any

# 引入一些常用处理器作为示例，你可以按需替换或自定义
from qlib.data.dataset.processor import Fillna, DropnaLabel

class MyCustomHandler(DataHandlerLP):
    """

    """

    def __init__(
        self,
        instruments: Union[str, List] = "csi300",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        freq: str = "day",  # 默认频率
        infer_processors: Optional[List[Processor]] = None,
        learn_processors: Optional[List[Processor]] = None,
        data_loader: Optional[QlibDataLoader] = None,
        **kwargs,
    ):
        

        # 1. 构建数据加载器 (Data Loader)
        # 如果用户没有传入自定义的 data_loader，则使用默认配置
        if data_loader is None:
            # 定义特征字段 (Features)
            feature_fields = [
                "$close / $open - 1",  # 开盘收益率
                # "$volume / ($high + 1e-9)", # 成交量/最高价 (避免除零)
                "$close",
            ]
            # 定义标签字段 (Labels) - 预测未来5天收益
            label_fields = ["Ref($close, -5) / $close - 1"]
            
            # 合并字段用于 DataLoader 配置 (QlibDataLoader 会自动区分 features 和 labels 如果配置得当，
            # 但通常我们直接传 config 给 features，label 单独传)
            # 注意：QlibDataLoader 的初始化签名通常是 config(即features), label, freq
            
            
            data_config = {
                "feature": feature_fields,  # 或者你的 feature 配置
                "label": label_fields       # 或者你的 label 配置
            }
            data_loader = QlibDataLoader(
                config=data_config,
                freq=freq               # 【关键】freq 在这里设置
            )

        # 2. 设置默认处理器 (如果未提供)
        if learn_processors is None:
            learn_processors = [
                Fillna(),
                # 可以在这里添加更多需要 "学习" 统计量的处理器，例如标准化
                # ZScoreNorm(fit_start_time, fit_end_time) 
            ]
        
        if infer_processors is None:
            infer_processors = [
                Fillna(),
            ]

        # 3. 准备传给父类的参数
        # 注意：fit_start_time 和 fit_end_time 是 DataHandlerLP 特有的，用于控制训练集统计量的计算范围
        # 它们需要从 kwargs 中提取（如果配置文件里写在 kwargs 里），或者直接使用函数参数
        
        # 确保从 kwargs 中更新 fit 时间，以防配置文件中是通过 **kwargs 传递的
        # 但在你的 YAML 配置中，它们是直接作为 kwargs 传给 class 的，所以函数参数已经接收到了。
        # 为了保险，我们可以检查是否还在 kwargs 里残留（通常不会，除非调用方式特殊）
        
        # 2. 准备传递给父类的参数
        # 显式列出父类 DataHandlerLP 需要的参数，不要直接用 **kwargs 包含多余参数
        parent_kwargs = {
            "instruments": instruments,
            "start_time": start_time,
            "end_time": end_time,
            "data_loader": data_loader,
            "infer_processors": infer_processors,
            "learn_processors": learn_processors,
        }
        
        # 将其他未知的合法参数加入 (排除掉 fit_start/end_time)
        # 确保 **kwargs 里没有 fit_start_time 等非法参数
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['fit_start_time', 'fit_end_time']}
        parent_kwargs.update(safe_kwargs)

        # 3. 调用父类初始化
        # 现在只传递父类认可的参数
        super().__init__(**parent_kwargs)

    # 如果需要完全自定义数据处理流程，可以重写 _process_data 等方法
    # 但对于大多数适配场景，配置好 Loader 和 Processor 即可