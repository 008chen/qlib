
from qlib.contrib.data.handler import Alpha158
import qlib
from qlib.constant import REG_CN

provider_uri = "~/.qlib/qlib_data/cn_data"
    # 数据存储路径
    # provider_uri = 'E:\\xiangmu\\qlib-main\\data\\qlib_bin\\qlib_bin'

qlib.init(provider_uri=provider_uri, region=REG_CN,kernels=1)
handler = Alpha158(
    start_time='2010-01-01',
    end_time='2020-12-31',
    fit_start_time='2010-01-01',
    fit_end_time='2015-12-31',
    instruments='csi300'
)
print("-------------------")
# 获取特征数据
features = handler.fetch(col_set='feature')
# 获取标签数据
labels = handler.fetch(col_set='label')

print('特征数据形状:', features.shape)
print('标签数据形状:', labels.shape)
