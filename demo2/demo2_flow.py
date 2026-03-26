import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R # 实验记录管理器
from qlib.workflow.record_temp import SignalRecord,SigAnaRecord,PortAnaRecord
from qlib.utils import flatten_dict

from qlib.data import D # 基础行情数据服务的对象

provider_uri = "~/.qlib/qlib_data/cn_data"  # ~表示系统默认的用户目录，通常是C盘Users目录下用户登录名子目录
# provider_uri = "G:/qlibtutor/qlib_data/cn_data"  # 原始行情数据存放目录

# 初始化, kernels=1，在计算特征表达式时只用一个核心，方便计算出错时进行调试。
# qlib.init(provider_uri=provider_uri, region=REG_CN, kernels=1)

# 初始化 QLib
qlib.init(provider_uri=provider_uri, region=REG_CN,kernels=1)
print("QLib 初始化成功！")

# print(D.features(['SH600000','SH600038'], fields=['$open','$high', '$low','$close','$change','$factor','$volume']))


# 定义股票池。
# stockpool的定义形式举例：stockpool='csi100', stockpool=D.instruments(market='csi100'),
# stockpool=['sh600000', 'sz000001']
stockpool =  D.instruments(market='csi300')

benchmark = "SH000300"  # 基准：沪深300指数

###################################
# 参数配置
###################################
# 数据处理器参数配置：整体数据开始结束时间，训练集开始结束时间，股票池
data_handler_config = {
    "start_time": "2008-01-01",   #  实际数据加载区间 - 最终返回的数据时间范围
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01", # 标准化参数的学习区间 - 用于计算均值、标准差等统计量的时间段
    "fit_end_time": "2014-12-31",
    "instruments": stockpool,
}

# 任务参数配置
task = {
    # 机器学习模型参数配置
    "model": {
        # 模型类
        "class": "LGBModel",
        # 模型类所在模块
        "module_path": "qlib.contrib.model.gbdt",
        # 模型类超参数配置，未写的则采用默认值。这些参数传给模型类
        "kwargs": {  # kwargs用于初始化上面的class
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
            "early_stopping_rounds": 50, # 训练迭代提前停止条件
            "num_boost_round": 1000, # 最大训练迭代次数
        },
    },
    "dataset": {  #　因子数据集参数配置
        # 数据集类，是Dataset with Data(H)andler的缩写，即带数据处理器的数据集
        "class": "DatasetH",
        # 数据集类所在模块
        "module_path": "qlib.data.dataset",
        # 数据集类的参数配置
        "kwargs": {
            "handler": { # 数据集使用的数据处理器配置
                "class": "Alpha158", # 数据处理器类，继承自DataHandlerLP
                "module_path": "qlib.contrib.data.handler", # 数据处理器类所在模块
                "kwargs": data_handler_config, # 数据处理器参数配置
            },
            "segments": { # 数据集时段划分
                "train": ("2008-01-01", "2014-12-31"), # 此时段的数据为训练集
                "valid": ("2015-01-01", "2016-12-31"), # 此时段的数据为验证集
                "test": ("2017-01-01", "2020-08-01"),  # 此时段的数据为测试集
            },
        },
    },
}

# 实例化模型对象
model = init_instance_by_config(task["model"])
# 实例化数据集，从基础行情数据计算出的包含所有特征（因子）和标签值的数据集。
dataset = init_instance_by_config(task["dataset"]) # 类型DatasetH

print("实例化数据集 初始化成功！")

# ---------------------------------------------------------执行训练模型实验--------------------------------------------
# R变量可以理解为实验记录管理器。
with R.start( experiment_name="train"): # 注意，设好实验名
    # 可选：记录task中的参数到运行记录下的params目录
    R.log_params(**flatten_dict(task))

    # 训练模型，得到训练好的模型model
    model.fit(dataset)

    # 可选：训练好的模型以pkl文件形式保存到本次实验运行记录目录下的artifacts子目录，以备后用
    R.save_objects(**{"trained_model.pkl": model})



    # 打印本次实验记录器信息，含记录器id，experiment_id等信息
    print('info,train', R.get_recorder().info)


    ########################################################################################################
    #                                  说明：
    # 一个实验（比如本实验train）对应mlruns下的一个实验id目录，例如1。
    # 一个实验的每次运行，会在该目录下生成一个不同的实验运行记录id子目录，例如65821e2597014122979f32fef465719f
    # 运行记录id目录中最重要的子目录是制品目录artifacts，里头保存了实验结果pkl文件
    #########################################################################################################



# ---------------------------预测：在测试集test上进行预测--------------------------------------------------------------
# 1. 执行预测实验
with R.start(experiment_name="predict"):

    # 当前实验的实验记录器：预测实验记录器
    predict_recorder = R.get_recorder()

    # 生成预测结果文件: pred.pkl, label.pkl存放在运行记录目录下的artifacts子目录
    # 本实验默认是站在t日结束时刻，预测t+2日收盘价相对t+1日的收益率，计算公式为 Ref($close, -2)/Ref($close, -1) - 1
    sig_rec = SignalRecord(model, dataset, predict_recorder)  # 将训练好的模型、数据集、预测实验记录器传递给信号记录器
    sig_rec.generate()



    # 生成预测结果分析文件，在artifacts\\sig_analysis 目录生成ic.pkl,ric.pkl文件
    sigAna_rec = SigAnaRecord(predict_recorder) # 信号分析记录器
    sigAna_rec.generate()

    print('info,predict', R.get_recorder().info)
    ###########################################################################
    #              说明
    # 由于定义了一个新实验名predict，所以mlruns目录中会新建一个实验id目录，例如2
    ###########################################################################



# 2.预测结果查询

# label_df = predict_recorder.load_object("label.pkl") # 这个pkl文件记录的是测试集未经数据预处理的原始标签值
# 测试集标签值，默认这是经过数据预处理比如标准化处理的（推理数据集的测试集部分）标签值
label_df = dataset.prepare("test", col_set="label")
label_df.columns = ['label'] # 修改列名LABEL0为label

pred_df = predict_recorder.load_object("pred.pkl") # 加载测试集预测结果到dataframe

print('label_df', label_df) # 预处理后的测试集标签值
print('pred_df', pred_df) # 测试集对标签的预测值，score就是预测值

# ----------------------------------信息系数IC和排序信息系数Rank IC--------------------------------------------
# 信息系数：每天根据所有股票的预测值和标签值，计算出二者在该日的相关系数，即为该日信息系数
ic_df = predict_recorder.load_object("sig_analysis/ic.pkl")
# 排序信息系数 rank ic：每天根据所有股票的预测值的排名和标签值的排名，计算出二者在该日的排序相关系数，即为该日排序信息系数
print('ic_df', ic_df)
ric_df = predict_recorder.load_object("sig_analysis/ric.pkl")
print('ric_df', ric_df)

print('list_metrics', predict_recorder.list_metrics()) # 所有绩效指标
print('IC', predict_recorder.list_metrics()['IC']) # IC均值：每日IC的均值，一般认为|IC|>0.03说明因子有效，注意 -0.05也认为有预测效能，说明负相关显著
print('ICIR', predict_recorder.list_metrics()['ICIR']) #IC信息率：平均IC/每日IC标准差,也就是方差标准化后的ic均值，一般而言，认为|ICIR|>0.6,因子的稳定性合格
print('Rank IC', predict_recorder.list_metrics()['Rank IC']) # 排序IC均值，作用类似IC
print('Rank ICIR', predict_recorder.list_metrics()['Rank ICIR']) # 排序IC信息率，作用类似ICIR# 此图用于评价因子单调性，组1是因子值最高的一组，组5是因子值最低的一组。

# 这里是评价的是score这个综合因子的有效性和稳定性
# 一般认为|IC|>0.03说明因子有效，|ICIR|>0.6,说明因子稳定


# ----------------------------------------预测绩效分析图----------------------------------------
#准备数据：测试集"
# 创建测试集"预测"和"标签"对照表
pred_label_df = pd.concat([pred_df, label_df], axis=1, sort=True).reindex(label_df.index)
pred_label_df


# --------信息系数ic 和 rank ic 图 （按天）
from qlib.contrib.report import analysis_position, analysis_model
analysis_position.score_ic_graph(pred_label_df)
# ic图形横坐标按天显示该日所有股票预测值和标签的相关系数
# 有时候，二者正相关，即预测值越大，则标签值也越大；预测越小，标签也越小。有时负相关，即预测越大，标签越小。有时相关性很小（相关系数接近0）。

# --------------------------模型绩效图
analysis_model.model_performance_graph(pred_label_df)
# 评价score这个综合因子，以下所说因子指score这个因子

# cumulative Return图
# 用于评价因子单调性，组1是因子值最高的一组，组5是因子值最低的一组。
# 若因子越大的组，收益率越高，说明因子单调性好，也就证明因子对收益率的预测越有效
# 各组收益率差异越大，说明因子特异性高，因子有效。一般看组1和组5的收益率差异是否大即可。

# IC分布图和 IC Normal Dist.Q-Q图
# 观察IC分布是否接近正太分布，越接近正太分布，说明因子越可靠。若ic均值挺大的，但是IC分布图极度但极度的尖峰或右偏，这样的情况，说明因子不可靠。

# Atuo Correlation图
# 评价因子自相关性
# 因子越是具有正的自相关性，则换手率越低，手续费也就越低，默认显示的是lag滞后一期的相关系数
# 如果因子自相关为0或负，则股票今天因子高，明天很可能因子就低，这样造成的结果就是，我们对这个股票，一会儿卖，一会儿买，从而造成很高的手续费。
analysis_model.model_performance_graph(pred_label_df, N=6,
    graph_names=["group_return", "pred_ic", "pred_autocorr",  "pred_turnover"],
    rank=True, lag=1, reverse=False, show_notebook=True) # N分几组,lag 自相关图滞后期

# top bottom turnover图
# 展示了1组（top）和5组（bottom）股票的换手率序列


# ----------------------------模型特征重要性
# 得到特征重要性系列
feature_importance = model.get_feature_importance()
print(feature_importance)
# feature_importance.plot(figsize=(50, 10))

fea_expr, fea_name = dataset.handler.get_feature_config() # 获取特征表达式，特征名字
# 特征名，重要性值的对照字典
feature_importance = {fea_name[int(i.split('_')[1])]: v for i,v in feature_importance.items()}
feature_importance

