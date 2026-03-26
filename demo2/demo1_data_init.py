import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R # 实验记录管理器
from qlib.workflow.record_temp import SignalRecord,SigAnaRecord,PortAnaRecord
from qlib.utils import flatten_dict

from qlib.data import D # 基础行情数据服务的对象

provider_uri = "~/.qlib/qlib_data/cn_data"  # ~表示系统默认的用户目录，通常是C盘Users目录下用户登录名子目录
# provider_uri = "d:/qlibtutor/qlib_
# 
# data/cn_data"  # 原始行情数据存放目录

# 初始化, kernels=1，在计算特征表达式时只用一个核心，方便计算出错时进行调试。
# qlib.init(provider_uri=provider_uri, region=REG_CN, kernels=1)

# 初始化 QLib
# qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
qlib.init(provider_uri=provider_uri, region=REG_CN,kernels=1)
print("QLib 初始化成功！")


# calendar = D.calendar()
# print(f"交易日历长度: {len(calendar)}")
# print(f"最近5个交易日: {calendar[-5:]}")


# print("加载指定时间范围和频率的交易日历")
# print(D.calendar(start_time='2010-01-01', end_time='2017-12-31', freq='day')[:2])



# print("将给定的市场名称解析为股票池配置")
# print(D.instruments(market='all'))

# print("在指定时间范围内加载特定股票池的标的")
# instruments = D.instruments(market='csi300')
# # 检索股票池代码，交易日历
# stockpool_list = D.list_instruments(instruments, as_list=True)
# print(stockpool_list)
# print(D.list_instruments(instruments=instruments, start_time='2010-01-01', end_time='2017-12-31', as_list=True)[:6])

# 基于股票代码列表定义股票池
# instruments = ['sh600000', 'sz000001'] 
# data_df = D.features(instruments, fields=['$open','$high', '$low','$close','$change','$factor','$volume'])
# print(data_df)



# print("根据名称过滤器从基础市场加载动态标的")
# from qlib.data.filter import NameDFilter
# nameDFilter = NameDFilter(name_rule_re='SH[0-9]{4}55')
# instruments = D.instruments(market='csi300', filter_pipe=[nameDFilter])
# print(D.list_instruments(instruments=instruments, start_time='2015-01-01', end_time='2016-02-15', as_list=True))



# print("根据表达式过滤器从基础市场加载动态标的")
# from qlib.data.filter import ExpressionDFilter
# expressionDFilter = ExpressionDFilter(rule_expression='$close>200')
# instruments = D.instruments(market='csi300', filter_pipe=[expressionDFilter])

# stockpool_list = D.list_instruments(instruments=instruments, start_time='2026-01-01', end_time='2026-03-15', as_list=True)
# data_df = D.features(stockpool_list, fields=['$open','$high', '$low','$close/$factor','$change','$factor','$volume'])
# print(data_df)


# print("在指定时间范围内加载特定股票池的特征")
# from qlib.data.filter import NameDFilter, ExpressionDFilter
# nameDFilter = NameDFilter(name_rule_re='SH[0-9]{4}55')
# expressionDFilter = ExpressionDFilter(rule_expression='$close>Ref($close,1)')
# instruments = D.instruments(market='csi300', filter_pipe=[nameDFilter, expressionDFilter])
# fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
# print(D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day').head().to_string())




# D.features(['SH600000','SH600038'], fields=['$open','$high', '$low','$close','$change','$factor','$volume'])



# 获取默认时间段的日线日历
calendar = D.calendar()
print(f"总交易日数: {len(calendar)}")
print(f"日期范围: {calendar[0]} 至 {calendar[-1]}")

# 获取指定时间段的日历
custom_calendar = D.calendar(start_time='2020-01-01', end_time='2020-12-31')
print(f"2020年交易日数: {len(custom_calendar)}")


# ------------------------从常用方法-------------------------------------------
# # 加载特征数据
# D.features(self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=None, inst_processors=[])

# # 加载股票池
# D.instruments(market='market_name')

# # 加载交易日历
# D.calendar(start_time=None, end_time=None, freq="day")

# # 列出股票池中的股票
# D.list_instruments(instruments, start_time=None, end_time=None, as_list=False)