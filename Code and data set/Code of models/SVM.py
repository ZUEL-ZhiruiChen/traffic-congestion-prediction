import pandas as pd
import numpy as np
from datetime import datetime
import warnings #运行这个代码可以让Python不显示warnings
from keras.models import Model
from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam ,Adamax
from keras.layers import Input, Dense, GRU, SimpleRNN
from keras import layers
import warnings
from keras import optimizers
from keras import initializers
from keras.utils import plot_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import sklearn
from tensorflow import keras  # 原来的代码为 import keras
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")

df = pd.read_csv('/Users/zipging/PycharmProjects/pythonProject6/xh_taxi_labelled.csv') ##读取之前处理好的数据
df['date'] = 20190924 #生成数据所在的日期
df['date'] = pd.to_datetime(df['date'],format='%Y%m%d')
df.fillna(0, inplace=True)
# 生成特征
def gerData(df, lag):
    jar = []
    for i in range(1, lag + 1):
        print(i)
        tmp = df.copy()
        tmp['time_id'] = tmp.time_id.values - i  # 每次循环生成t-i时间步的数据
        tmp = tmp.set_index(['rowid', 'colid', 'time_id', 'date'])
        jar.append(tmp)
    # 将各时间步数据进行拼接
    jar.append(df[['rowid', 'colid', 'time_id', 'date', 'labels']].set_index(['rowid', 'colid', 'time_id', 'date']))

    return pd.concat(jar, axis=1).reset_index()
used = ['rowid', 'colid', 'time_id', 'aveSpeed', 'gridAcc', 'speed_std', 'labels', 'date']
dataRaw = df[used]  # 筛选标签
data = gerData(dataRaw, 6)  # 生成训练数据
data = data.dropna()  # 去除空值

#data:需要进行分割的数据集
#random_state:设置随机种子，保证每次运行生成相同的随机数
line_sen_list =  data.values[:, 4:-1]
label_list = data.values[:, -1]
# 先将1.训练集，2.验证集+测试集，按照6：4进行随机划分
train_x, X_validate_test, train_y, y_validate_test = train_test_split(line_sen_list, label_list, test_size = 0.4, random_state = 42)
# 再将1.验证集，2.测试集，按照1：1进行随机划分
val_x, test_x, val_y, test_y = train_test_split(X_validate_test, y_validate_test, test_size = 0.5, random_state = 42)

# 将数据转为浮点型
train_x= train_x.astype(np.float32)
test_x=test_x.astype(np.float32)
train_y=train_y.astype(np.float32)
test_y=test_y.astype(np.float32)
val_x=val_x.astype(np.float32)
val_y=val_y.astype(np.float32)

# 去除异常值
mins = [np.nanmin(train_x[:, i][train_x[:, i] != -np.inf]) for i in range(train_x.shape[1])]
maxs = [np.nanmax(train_x[:, i][train_x[:, i] != np.inf]) for i in range(train_x.shape[1])]

# 一次遍历矩阵的一列，替换+和-无穷
# 该列的最大值或最小值
for i in range(train_x.shape[1]):
    train_x[:, i][train_x[:, i] == -np.inf] = mins[i]
    train_x[:, i][train_x[:, i] == np.inf] = maxs[i]

mins = [np.nanmin(val_x[:, i][val_x[:, i] != -np.inf]) for i in range(val_x.shape[1])]
maxs = [np.nanmax(val_x[:, i][val_x[:, i] != np.inf]) for i in range(val_x.shape[1])]

# 一次遍历矩阵的一列，替换+和-无穷大
# 该列的最大值或最小值
for i in range(val_x.shape[1]):
    val_x[:, i][val_x[:, i] == -np.inf] = mins[i]
    val_x[:, i][val_x[:, i] == np.inf] = maxs[i]

mins = [np.nanmin(test_x[:, i][test_x[:, i] != -np.inf]) for i in range(test_x.shape[1])]
maxs = [np.nanmax(test_x[:, i][test_x[:, i] != np.inf]) for i in range(test_x.shape[1])]

# 一次遍历矩阵的一列，替换+和-无穷大
# 该列的最大值或最小值
for i in range(val_x.shape[1]):
    test_x[:, i][test_x[:, i] == -np.inf] = mins[i]
    test_x[:, i][test_x[:, i] == np.inf] = maxs[i]

# 将输入归一化为0和1之间
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

# 转换格式,对于决策树等不支持tensor格式的模型则删去该部分
train_x = train_x.reshape(-1, 6, 4)
val_x = val_x.reshape(-1, 6, 4)
test_x = test_x.reshape(-1, 6, 4)

# 进行onehot编码
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0], [1], [2]])
train_y = train_y.reshape(-1, 1)
train_y = encoder.transform(train_y).toarray()
test_y = test_y.reshape(-1, 1)
test_y = encoder.transform(test_y).toarray()
val_y = val_y.reshape(-1, 1)
val_y = encoder.transform(val_y).toarray()

# 将训练集和验证集合并
train_x_con = np.concatenate([train_x, val_x], axis=0)
train_y_con = np.concatenate([train_y, val_y], axis=0)
train_x_con = train_x_con.reshape(-1, 24)
test_x_con = test_x.reshape(-1, 24)
train_y_con = train_y_con.argmax(axis=1)

# 决策树
clf = sklearn.svm.LinearSVC()
clf.fit(train_x_con,train_y_con) #进⾏模型拟合
pred = clf._predict_proba_lr(test_x_con)
pred = np.array(pred)
print(sklearn.metrics.classification_report(test_y.argmax(axis=1), pred.argmax(axis=1),  digits=6)) #进⾏测试集测试
sklearn.metrics.roc_auc_score(test_y.argmax(1), pred, average='weighted', multi_class='ovo')