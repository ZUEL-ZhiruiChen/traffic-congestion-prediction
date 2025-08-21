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
from keras.layers import Input, Dense, GRU
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
# 进入tensorflow_backend.py文件： 2.修改； 3.添加
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
train_set, test_set_validate = train_test_split(data, test_size=0.4, random_state=42)
valid_set, test_set = train_test_split(test_set_validate, test_size=0.5, random_state=42)

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


def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)  # 第一卷积
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)  # 第二卷积
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  # shortcut（捷径）
    o = add([r, shortcut])
    o = Activation('relu')(o)
    #o = BatchNormalization()(o)
    return o

# TCN部分
def TCN(x):
    #inputs = Input(x)
    x = ResBlock(x, filters=16, kernel_size=3, dilation_rate=1)
    x = ResBlock(x, filters=16, kernel_size=3, dilation_rate=2)
    x = ResBlock(x, filters=8, kernel_size=3, dilation_rate=4)
    return x

# 位置矩阵Attention环节
def DMAttention(x):
    gamma = tf.Variable(tf.ones(1), name='gamma', trainable=True)
    x_origin = x
    proj_query, proj_key, proj_value = x, x, x
    #proj_query = Dense(proj_query.shape[2])(proj_query)
    #proj_key = Dense(proj_key.shape[2])(proj_key)
    #proj_value = Dense(proj_value.shape[2])(proj_value)
    proj_key = tf.transpose(proj_key, perm=[0, 2, 1]) # 对k进行转置,q和v不动
    energy = tf.matmul(proj_query, proj_key)
    attention = Activation('softmax')(energy)
    #attention = tf.nn.softmax(energy, name='attention')
    proj_value = tf.transpose(proj_value, perm=[0, 2, 1]) #对v进行转置
    out = tf.matmul(proj_value, attention)
    out = tf.transpose(out, perm=[0, 2, 1])
    out = add([out*gamma, x_origin])
    out = BatchNormalization()(out)
    return out


class DM(Layer):
    '''
    Dense layer
    '''

    def __init__(self, weights=None, axis=-1, beta_init = 'zero', gamma_init = 'one', momentum = 0.9, **kwargs):
        # 初始化权重参数
        # 权重w占位符
        self.gamma_init = initializers.Ones()
        super(DM, self).__init__(**kwargs)
    def build(self, input_shape):

        # Compatibility with TensorFlow >= 1.0.0
        self.gamma = K.variable(self.gamma_init(input_shape), name='{}_gamma'.format(self.name))

    # 通过回调函数计算
    def call(self, input, *args, **kwargs):
        x_origin = inputs
        proj_query = inputs, inputs
        proj_key = inputs
        proj_value = inputs
        proj_key = tf.transpose(proj_key, perm=[0, 2, 1])  # 对k进行转置,q和v不动
        energy = tf.matmul(proj_query, proj_key)
        attention = Activation('softmax')(energy)
        proj_value = tf.transpose(proj_value, perm=[0, 2, 1])  # 对v进行转置
        out = tf.matmul(proj_value, attention)
        out = tf.transpose(out, perm=[0, 2, 1])
        out = add([out * self.gamma, x_origin])
        out = BatchNormalization()(out)
        return out  # matmul 矩阵乘法

def multi_category_focal_loss1(alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
    当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    #alpha = tf.constant([[1],[1],[1],[1],[1]], dtype=tf.float32)
    #alpha = tf.constant_initializer(alpha)
    gamma = float(gamma)
    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss1_fixed

class Linear(layers.Layer):
    '''
    为编写树突网络，先编写线性层
    '''
    def __init__(self, time_step, input_dim):
        super(Linear, self).__init__()
        # 初始化权重参数
        w_init = tf.random_normal_initializer()
        # 权重w占位符
        self.w = tf.Variable(initial_value=w_init(shape=(time_step, input_dim),
                                                  dtype='float32'),
                             trainable=True)

    # 通过回调函数计算
    def call(self, inputs):
        return tf.matmul(inputs, self.w) # matmul 矩阵乘法


class ddNet(layers.Layer):

    def __init__(self):
        super(ddNet, self).__init__()

        # xw+b
        self.fc0 = Linear(16, 16)
        self.dd = Linear(16, 16)
        self.dd2 = Linear(16, 16)
        self.dd3 = Linear(16, 16)
        self.dd4 = Linear(16, 16)
        self.fc2 = Linear(16, 16)

    def call(self, x):
        # x: [b, 1, 28, 28]
        x = self.fc0(x)
        c = x
        # h1 = x@w1*x

        x = self.dd(x) * c
        x = self.dd2(x) * c
        x = self.dd3(x) * c
        x = self.dd4(x) * c

        x = self.fc2(x)
        return x



adam = Adam(lr=0.0005)
output_dim = 3
# 每轮训练模型时，样本的数量
batch_size = 256
# 训练400轮次
epochs = 20
time_step = 6
input_dim = 4

pre = np.array(range(20)).astype(np.float32)
f1 = np.array(range(20)).astype(np.float32)
acc = np.array(range(20)).astype(np.float32)
auc = np.array(range(20)).astype(np.float32)


tf.random.set_seed(42)
input = Input(shape=(time_step, input_dim))
# 进行TCN

x = Dense(16)(input)
x = Dense(16)(x)

# 做FC并构建模型
output = Flatten()(x)
output = Dense(output_dim, activation='softmax')(output)
model = Model(inputs=input, outputs=output)
# 模型编译
model.compile(loss=multi_category_focal_loss1(alpha=[[38405],[9410],[39824]], gamma=2), optimizer=adam, metrics=['accuracy'])  # 编译模型
#model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])  # 编译模型
result = model.fit(train_x.astype(np.float32), train_y.astype(np.float32), epochs=epochs,
                   validation_data=(train_x.astype(np.float32), train_y.astype(np.float32)), batch_size=batch_size)
#result = model.fit(train_x.astype(np.float32), train_y.astype(np.float32), epochs=epochs, validation_data=(val_x.astype(np.float32), val_y.astype(np.float32)), batch_size=batch_size)  # 训练模型
#test_loss = model.evaluate(test_x, test_y)  # 评估模型
y_pred = model.predict(test_x)
history = result.history
    #print(sklearn.metrics.classification_report(test_y.argmax(1), y_pred.argmax(1), digits=6)) #输出报告
sklearn.metrics.classification_report(test_y.argmax(1), y_pred.argmax(1), digits=6, output_dict=True)
 pd.DataFrame(m)
pre[i] = m['weighted avg'][0]
f1[i] = m['weighted avg'][2]
acc[i] = m['accuracy'][0]
auc[i] = sklearn.metrics.roc_auc_score(test_y.argmax(1), y_pred, average='weighted', multi_class='ovo')

# 预测结果可视化
y_hat = model.predict(test_x)

label_hat = np.argmax(y_hat, axis=1)

predResult = np.hstack((test_set[['rowid', 'colid']].values, label_hat.reshape(-1, 1), test_y.argmax(1).reshape(-1, 1)))

predResult = predResult.astype(int)

rows = predResult[:, 0].max() + 1
cols = predResult[:, 1].max() + 1
matPred = np.ones((rows, cols)) + 2
matTrue = np.ones((rows, cols)) + 2

for i in range(len(predResult)):
    row = predResult[i][0]
    col = predResult[i][1]
    matPred[row, col] = predResult[i][2]
    matTrue[row, col] = predResult[i][3]

matPred = matPred.astype(int)
matTrue = matTrue.astype(int)

a1 = pd.DataFrame(matPred)
a2 = pd.DataFrame(matTrue)
a1.to_csv('matPred.csv')
a2.to_csv('matTrue.csv')
###结果对比
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
#ft = matplotlib.font_manager.FontProperties(fname='/home/mw/input/TaxiData7578/NotoSansCJK.ttc', size=14)
#plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Songti SC']

sns.set(font_scale=2)
cmap = 'GnBu'
ticks = range(5)

##真实值
plt.figure(figsize=(10, 6))
sns.heatmap(matTrue, cmap=cmap, cbar_kws={"ticks":ticks}, xticklabels=False, yticklabels=False)
#ax2.set_title('真实值')
plt.title('Actual value', fontsize=18)
##预测值
plt.figure(figsize=(20, 6))
sns.heatmap(matPred, cmap=cmap,  cbar_kws={"ticks":ticks}, xticklabels=False, yticklabels=False)
plt.title('Predicted value', fontsize=18)
plt.show()