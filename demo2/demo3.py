
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R # 实验记录管理器
from qlib.workflow.record_temp import SignalRecord,SigAnaRecord,PortAnaRecord
from qlib.utils import flatten_dict

from qlib.data import D # 基础行情数据服务的对象
from qlib.data.ops import EMA

if __name__ == '__main__':
    
    provider_uri = "~/.qlib/qlib_data/cn_data"
    # 数据存储路径
    # provider_uri = 'E:\\xiangmu\\qlib-main\\data\\qlib_bin\\qlib_bin'

    qlib.init(provider_uri=provider_uri, region=REG_CN,kernels=1)
    # 获取沪深 300 指数成分股
    csi300_instruments = D.instruments(market='csi300')
    csi300_stocks = D.list_instruments(instruments=csi300_instruments, as_list=True)
    print(f"CSI300成分股数量: {len(csi300_stocks)}")
  
  
    ema12 = EMA('$close', 12)
    
    # 获取行情数据
    # 二索引dataframe
    df = D.features(
        instruments=csi300_stocks,
        fields=['$open', '$close', '$high', '$low', '$volume',ema12],
        start_time='2010-01-01',
        end_time='2020-12-31',
        freq='day'
    )
    print(df.head())
