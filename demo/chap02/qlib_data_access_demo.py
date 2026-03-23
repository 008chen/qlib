import qlib
from qlib.data import D
import pandas as pd

# 初始化Qlib（请根据实际数据路径修改）
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

# 获取交易日历
days = D.calendar(start_time="2020-01-01", end_time="2020-01-10", freq="day")
print("交易日历:", days)

# 获取CSI300成分股列表（返回股票池配置字典）
instruments_conf = D.instruments('csi300')
# 获取实际股票代码列表
inst_list = D.list_instruments(instruments_conf, start_time="2020-01-01", end_time="2020-01-10", freq="day", as_list=True)
print("CSI300成分股样例:", inst_list[:5])

# 获取特征数据
data = D.features(
    instruments=[inst_list[0]],
    fields=["$close", "$volume", "$open"],
    start_time="2020-01-01",
    end_time="2020-01-10",
    freq="day"
)
print("特征数据样例:")
print(data) 