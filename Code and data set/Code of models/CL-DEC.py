import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import math
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from keras.layers import add
from keras.layers import  Conv1D, Activation, Flatten, Dense
from sklearn.preprocessing import Normalizer
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras import layers
import warnings
from keras import optimizers
from keras import initializers
from keras.utils import plot_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers import Input, Lambda
from keras import backend as K
import os
from keras.optimizers import RMSprop, SGD, Nadam
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
import os
import time
warnings.filterwarnings("ignore")

def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    y_true只是凑数的，并不起作用。因为真正的y_true是通过batch内数据计算得出的。
    y_pred就是batch内的每个样本的embedding，通过定义的encoder得来
    """
    # 构造标签
    # idxs = [0,1,2,3,4,5]
    idxs = K.arange(0, K.shape(y_pred)[0])
    # 给idxs添加一个维度，idxs_1 = [[0,1,2,3,4,5]]
    idxs_1 = idxs[None, :]
    # 获取每个样本的同正例id，即
    idxs_2 = ((idxs+tf.cast(K.shape(y_pred)[0]/2,dtype = tf.int32))%(K.shape(y_pred)[0]))[:, None]
    # 生成计算loss时可用的标签
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    # 首先对样本向量各个维度做了一个L2正则，使其变得各项同性，避免下面计算相似度时，某一个维度影响力过大。
    y_pred = K.l2_normalize(y_pred, axis=1)
    # 其次，计算batch内每个样本的内积相似度(其实就是余弦相似度)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    # 然后，将矩阵的对角线部分变为0，代表样本自身不参与
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    # 最后，将所有相似度乘以20，这个目的是想计算softmax概率时，更加有区分度
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# 对比学习实验
def encoder_part(input):
    output1 = Dense(8, activation='relu')(input)
    output2 = Dense(32, activation='relu')(output1)
    output3 = Dense(64, activation='relu')(output2)
    output4 = Dense(128, activation='relu')(output3)
    output5 = Dense(8, activation='relu')(output4)
    return output5

def decoder_part(input):
    output = Dense(128, activation='relu')(input)
    output = Dense(64, activation='relu')(output)
    output = Dense(32, activation='relu')(output)
    output = Dense(8, activation='relu')(output)
    output = Dense(8, activation='relu')(output)
    output = Dense(2, activation='relu')(output)
    return output

grib_data = pd.read_csv('/Users/zipging/PycharmProjects/pythonProject4/data_xhtaxi.csv')
grib_data = grib_data.dropna()
grib_data.index = range(len(grib_data))
# 选择特征
feature = ['stopNum', 'aveSpeed']

# 数据标准化
scaler = MinMaxScaler()
scaler.fit(grib_data[feature])
grib_data_nor = scaler.transform(grib_data[feature])

# 通过对比学习进行特征提取
tf.random.set_seed(42)
bath_size = 20000
epochs = 100
inputs = Input(shape=(2))
lay1 = Dense(8, activation='relu')(inputs)
# 使用Dropout进行数据增强
sample1 = Dropout(0.1, seed=3)(lay1)
sample2 = Dropout(0.1, seed=4)(lay1)
# encoder部分
sample1_1 = encoder_part(sample1)
sample2_1 = encoder_part(sample2)
sample1_2 = add([sample1_1, lay1])
sample2_2 = add([sample2_1, lay1])
encoder = sample1_2
# decoder部分
sample1 = decoder_part(sample1_2)
sample2 = decoder_part(sample2_2)
# 将样本合并
result = tf.concat([sample1, sample2], axis=0)
output = result
contrast_learning = Model(inputs=inputs, outputs=sample1)
contrast_learning.compile(loss='mse', optimizer='adam')
history = contrast_learning.fit(grib_data_nor, grib_data_nor,validation_split=0.25, epochs=epochs, batch_size=bath_size, shuffle=False)
#df1 = contrast_learning.predict(grib_data_nor)
encoder = Model(inputs=inputs, outputs=encoder)

# 聚类部分
n_clusters = 3
from keras.models import load_model
model = load_model('还能看看的模型.h5')
encoder = Model(inputs=model.input, outputs=model.get_layer('add_46').output)
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)
# 使用K-means进行聚类中心初始化
kmeans = KMeans(n_clusters=n_clusters, n_init=200)
y_pred = kmeans.fit_predict(encoder.predict(grib_data_nor))
y_pred_last = np.copy(y_pred)
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# 定义T-分布函数
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# 进行迭代训练
model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
maxiter = 1000
update_interval = 140
y = None
batch_size = 128
loss = 0
index = 0
index_array = np.arange(grib_data_nor.shape[0])
tol = 0.01 # earlystop的容忍度
tf.random.set_seed(43)
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(grib_data_nor, verbose=1)
        # 更行辅助分布p
        p = target_distribution(q)
        # 评估聚类表现
        y_pred = q.argmax(1)
        # 模型收敛判别
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, grib_data_nor.shape[0])]
    model.train_on_batch(x=grib_data_nor[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= grib_data_nor.shape[0] else 0

# 可视化聚类结果
grib_data_labelled = pd.concat((grib_data,
                         pd.DataFrame(y_pred, columns = ['labels'])),
                         axis=1)
# 可视化聚类结果
plt.rcParams['font.sans-serif'] = ['Songti SC']
grib_df = pd.read_csv('/Users/zipging/PycharmProjects/pythonProject6/就决定是你了！.csv')
 # 判断各簇实际意义
grouped = grib_df.groupby(['labels']).mean()['aveSpeed']
congested = int(grouped.idxmin(axis=0))
clear = int(grouped.idxmax(axis=0))
slow = [x for x in [0,1,2] if x not in [congested, clear]][0]
# 0为缓行，2为拥堵，1为畅通
# 绘图
fig = plt.figure(figsize=(8,6), dpi=300)
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
plt.scatter(grib_df[grib_df['labels']==slow]['aveSpeed'], grib_df[grib_df['labels']==slow]['stopNum'],
            label='Slow', color='y', alpha=0.5)
plt.scatter(grib_df[grib_df['labels']==clear]['aveSpeed'], grib_df[grib_df['labels']==clear]['stopNum'],
            label='Clear', color='g', alpha=0.5)
plt.scatter(grib_df[grib_df['labels']==congested]['aveSpeed'], grib_df[grib_df['labels']==congested]['stopNum'],
            label='Congested', color='r', alpha=0.5)
plt.tick_params(labelsize=18)
plt.xlabel('Speed',fontsize=20)
plt.ylabel('StopNum',fontsize=20)
plt.legend(fontsize=18, loc='upper right')
plt.savefig('聚类结果', dpi=1000, bbox_inches='tight')
plt.show()
#grib_df = grib_data_labelled
