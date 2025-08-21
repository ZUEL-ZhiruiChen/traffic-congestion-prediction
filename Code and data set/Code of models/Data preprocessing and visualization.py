import warnings #运行这个代码可以让Python不显示warnings
import pandas as pd
import numpy as np
import time # 导入time库，时间戳
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import mapclassify
from shapely.geometry import Point,Polygon,shape
warnings.filterwarnings("ignore")

# 导入数据
df = pd.read_csv('/Users/zipging/Downloads/论文/论文2/190924.csv', header=None)
df = df.iloc[1:, [1, 3, 4, 5, 8, 6]]
# 给数据命名列, '车号','时间','经度','纬度','空车（1为载客，0为空车）','速度'
df.columns = ['VehicleNum', 'Stime', 'Lng', 'Lat', 'OpenStatus', 'Speed']
df.index = range(len(df))
df.head(10)#查看前10条数据

# 删除异常轨迹
df = df.sort_values(by = ['VehicleNum','Stime'])# 根据车牌号和时间进行排序
# 排序之后一段OD轨迹应为都是0或都是1，但数据中有可能出现前后都是0，突然有一条数据变成1，
# 或者前后都是1，突然变成0。对异常情况进行排除
df = df[-((df['OpenStatus'].shift() == df['OpenStatus'].shift(-1))&\
(df['OpenStatus'].shift(-1) != df['OpenStatus'])&\
(df['VehicleNum'].shift(-1) == df['VehicleNum'])&\
(df['VehicleNum'].shift() == df['VehicleNum']))]

# 删去不属于杭州范围的数据
# 杭州的大致经纬度范围
lon1 = 118.339684
lon2 = 120.714013
lat1 = 29.188585
lat2 = 30.565108
df[['Lng','Lat']] = df[['Lng','Lat']].apply(pd.to_numeric)
df_sz = df[(df.iloc[:, 2]>=lon1)&(df.iloc[:, 2]<=lon2)&(df.iloc[:, 3]>=lat1)&(df.iloc[:, 3]<=lat2)]

# 数据OD处理
#创建一列OpenStatus1，它的值是OpenStatus整体往上移一行
df_sz[['OpenStatus', 'Speed']] = df_sz[['OpenStatus', 'Speed']].apply(pd.to_numeric)
df_sz['OpenStatus1'] = df_sz['OpenStatus'].shift(-1)
#创建一列StatusChange，它的值是OpenStatus1减去OpenStatus，表示载客状态的变化
df_sz['StatusChange'] = df_sz['OpenStatus1']-df_sz['OpenStatus']
#提取其中的OD信息
oddata = df_sz[(df_sz['StatusChange']==1)|(df_sz['StatusChange']==-1)&\
(df_sz['VehicleNum'] == df_sz['VehicleNum'].shift(-1))]
oddata = oddata[['VehicleNum', 'Stime', 'Lng', 'Lat', 'StatusChange']]
oddata.columns = ['VehicleNum', 'Stime', 'SLng', 'SLat', 'StatusChange']
oddata['Etime'] = oddata['Stime'].shift(-1)
oddata['ELng'] = oddata['SLng'].shift(-1)
oddata['ELat'] = oddata['SLat'].shift(-1)
oddata = oddata[(oddata['VehicleNum'] == oddata['VehicleNum'].shift(-1))&\
(oddata['StatusChange'] == 1)]
oddata = oddata.drop('StatusChange',axis = 1)
oddata.head()
##Stime 为起点的时间，Etime为终点的时间，也就是乘客下车的时间
##SLng SLat为起点的经纬度，ELng Elat为终点的经纬度

# 读取杭州市底图
sz = gpd.GeoDataFrame.from_file('/Users/zipging/Downloads/论文/论文2/杭州底图.json',encoding = 'utf-8')
sz.plot()
plt.show()

# 栅格化代码
import math

# 定义一个测试栅格划的经纬度
testlon = 119.0
testlat = 29.5

# 划定栅格划分范围
lon1 = 118.339684
lon2 = 120.714013
lat1 = 29.188585
lat2 = 30.565108

latStart = min(lat1, lat2)
lonStart = min(lon1, lon2)

# 定义栅格大小(单位m)
accuracy = 400

# 计算栅格的经纬度增加量大小Lon和Lat
deltaLon = accuracy * 360 / (2 * math.pi * 6371004 * math.cos((lat1 + lat2) * math.pi / 360))
deltaLat = accuracy * 360 / (2 * math.pi * 6371004)

# 计算栅格的经纬度编号
LONCOL = divmod(float(testlon) - (lonStart - deltaLon / 2), deltaLon)[0]
LATCOL = divmod(float(testlat) - (latStart - deltaLat / 2), deltaLat)[0]

# 计算栅格的中心点经纬度
# HBLON = LONCOL*deltaLon + (lonStart - deltaLon / 2)#格子编号*格子宽+起始横坐标-半个格子宽=格子中心横坐标
# HBLAT = LATCOL*deltaLat + (latStart - deltaLat / 2)
# 以下为更正，不需要减去半个格子宽
HBLON = LONCOL * deltaLon + lonStart  # 格子编号*格子宽+起始横坐标=格子中心横坐标
HBLAT = LATCOL * deltaLat + latStart

# 定义空的geopandas表
data = gpd.GeoDataFrame()

# 定义空的list，后面循环一次就往里面加东西
LONCOL = []
LATCOL = []
geometry = []
HBLON1 = []
HBLAT1 = []

# 计算总共要生成多少个栅格
# lon方向是lonsnum个栅格
lonsnum = int((lon2 - lon1) / deltaLon) + 1
# lat方向是latsnum个栅格
latsnum = int((lat2 - lat1) / deltaLat) + 1

for i in range(lonsnum):
    for j in range(latsnum):
        HBLON = i * deltaLon + lonStart
        HBLAT = j * deltaLat + latStart
        # 把生成的数据都加入到前面定义的空list里面
        LONCOL.append(i)
        LATCOL.append(j)
        HBLON1.append(HBLON)
        HBLAT1.append(HBLAT)

        # 生成栅格的Polygon形状
        HBLON_1 = (i + 1) * deltaLon + lonStart
        HBLAT_1 = (j + 1) * deltaLat + latStart
        geometry.append(Polygon([
            (HBLON - deltaLon / 2, HBLAT - deltaLat / 2),
            (HBLON_1 - deltaLon / 2, HBLAT - deltaLat / 2),
            (HBLON_1 - deltaLon / 2, HBLAT_1 - deltaLat / 2),
            (HBLON - deltaLon / 2, HBLAT_1 - deltaLat / 2)]))

# 为geopandas文件的每一列赋值为刚刚的list
data['LONCOL'] = LONCOL
data['LATCOL'] = LATCOL
data['HBLON'] = HBLON1
data['HBLAT'] = HBLAT1
data['geometry'] = geometry

# 筛选出杭州范围的栅格
grid_sz = data[data.intersects(sz.unary_union)]
grid_sz.plot()
plt.savefig('栅格化蓝底图', dpi=1000, bbox_inches='tight')
plt.show()

#将OD数据并入栅格中
#计算od起终点所属的栅格编号
oddata = oddata[-oddata['Etime'].isnull()]
oddata['SLONCOL'] = ((oddata['SLng']-(lonStart - deltaLon / 2))/deltaLon).astype('int')
oddata['SLATCOL'] = ((oddata['SLat']-(latStart - deltaLat / 2))/deltaLat).astype('int')
oddata['ELONCOL'] = ((oddata['ELng']-(lonStart - deltaLon / 2))/deltaLon).astype('int')
oddata['ELATCOL'] = ((oddata['ELat']-(latStart - deltaLat / 2))/deltaLat).astype('int')
#集计
oddata = oddata.groupby(['SLONCOL','SLATCOL','ELONCOL','ELATCOL'])['VehicleNum'].count().rename('count').reset_index()
#筛选范围内的栅格
oddata = oddata[(oddata['SLONCOL']>=0) & (oddata['SLONCOL']<=lonsnum)&\
(oddata['SLATCOL']>=0) & (oddata['SLATCOL']<=latsnum)&\
(oddata['ELONCOL']>=0) & (oddata['ELONCOL']<=lonsnum)&\
(oddata['ELATCOL']>=0) & (oddata['ELATCOL']<=latsnum)]
#计算栅格的中心点经纬度
oddata['SHBLON'] = oddata['SLONCOL']*deltaLon + lonStart#格子编号*格子宽+起始横坐标=格子中心横坐标
oddata['SHBLAT'] = oddata['SLATCOL']*deltaLat + latStart
oddata['EHBLON'] = oddata['ELONCOL']*deltaLon + lonStart#格子编号*格子宽+起始横坐标=格子中心横坐标
oddata['EHBLAT'] = oddata['ELATCOL']*deltaLat + latStart

#将oddata转换为geodataframe，并生成geometry
oddata = gpd.GeoDataFrame(oddata)
from shapely.geometry import LineString
r = oddata.iloc[0]
oddata['geometry']=oddata.apply(lambda r:LineString([[r['SHBLON'],r['SHBLAT']],[r['EHBLON'],r['EHBLAT']]]),axis = 1)
oddata.plot()
plt.show()

##绘制OD分布图
#创建图框
fig = plt.figure(1,(16,8),dpi = 300)
ax = plt.subplot(111)
plt.sca(ax)

#绘制栅格
grid_sz.plot(ax = ax,edgecolor = (0,0,0,0.8),facecolor = (0,0,0,0),linewidths=0.02)

#绘制深圳的边界
sz_all = gpd.GeoDataFrame()
sz_all['geometry'] = [sz.unary_union]
sz_all.plot(ax = ax,edgecolor = (0,0,0,1),facecolor = (0,0,0,0),linewidths=0.5)
#设置字体 防止乱码
#ft = mpl.font_manager.FontProperties(fname='/Users/zipging/Downloads/论文/论文2/NotoSansCJK-Regular.ttc', size=11)
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False
plt.title('深圳市出租车OD分布', fontsize=18, verticalalignment='center', horizontalalignment= 'center')
#设置colormap的数据
vmax = oddata['count'].max()
cmapname = 'cool'
cmap = mpl.cm.get_cmap(cmapname)

#绘制OD
oddata.plot(ax = ax,column = 'count',linewidth = 15*(oddata['count']/oddata['count'].max()),cmap = cmap,vmin =0,vmax = vmax,alpha=0.5)

plt.axis('off')
#设定显示范围
ax.set_xlim(118.0,121.0)
ax.set_ylim(29.0,31.0)

#绘制colorbar
plt.imshow([[0,vmax]], cmap=cmap)
cax = plt.axes([0.08, 0.4, 0.02, 0.3])
plt.colorbar(cax=cax)

#保存图片
plt.savefig('杭州市出租车OD分布图', dpi=1000, bbox_inches='tight')
#显示图
plt.show()

##绘制OD的栅格分布
#以OD起点集计oddata
odcount = oddata.groupby(['SLONCOL','SLATCOL'])['count'].sum().reset_index()
#重命名列以便后面的表连接
odcount.columns = ['LONCOL','LATCOL','count']
#表连接
odcount_grid = pd.merge(grid_sz,odcount,on = ['LONCOL','LATCOL'])

#绘制O点分布图
# import matplotlib
# import matplotlib.pyplot as plt
fig = plt.figure(1,(16,8),dpi = 300)
ax = plt.subplot(111)
plt.sca(ax)

#设置colormap的数据
# import matplotlib
vmax = odcount_grid['count'].max()
cmapname = 'RdYlBu_r'
cmap = mpl.cm.get_cmap(cmapname)

#绘制栅格
odcount_grid.plot(ax = ax,column = 'count',scheme='quantiles',legend=False,cmap='RdYlBu_r',k=9, edgecolor='none',linewidth=0.3,
       legend_kwds={'loc':4,})

#绘制杭州边界
sz_all = gpd.GeoDataFrame()
sz_all['geometry'] = [sz.unary_union]
sz_all.plot(ax = ax,edgecolor = (0,0,0,1),facecolor = (0,0,0,0),linewidths=0.5)
#绘制标题
plt.axis('off')
plt.title('Hangzhou city taxi origin point distribution', fontsize=18, verticalalignment='center', horizontalalignment= 'center')
ax.set_xlim(118.0,121.0)
ax.set_ylim(29.0,31.0)
#绘制colorbar
plt.imshow([[0,vmax]], cmap=cmap)
cax = plt.axes([0.08, 0.4, 0.02, 0.3])
plt.colorbar(cax=cax)
##保存图片
#plt.savefig(r'深圳市出租车打车点分布.jpg',dpi=300)
plt.savefig('杭州市出租车打车点', dpi=1000, bbox_inches='tight')
plt.show()

#绘制D点分布图
# import matplotlib
# import matplotlib.pyplot as plt
#以OD起终点集计oddata
odcount = oddata.groupby(['ELONCOL','ELATCOL'])['count'].sum().reset_index()
#重命名列以便后面的表连接
odcount.columns = ['LONCOL','LATCOL','count']
#表连接
odcount_grid = pd.merge(grid_sz,odcount,on = ['LONCOL','LATCOL'])

fig = plt.figure(1,(16,8),dpi = 300)
ax = plt.subplot(111)
plt.sca(ax)

#设置colormap的数据
# import matplotlib
vmax = odcount_grid['count'].max()
cmapname = 'RdYlBu_r'
cmap = mpl.cm.get_cmap(cmapname)

#绘制栅格
odcount_grid.plot(ax = ax,column = 'count',scheme='quantiles',legend=False,cmap='RdYlBu_r',k=9, edgecolor='none',linewidth=0.3,
       legend_kwds={'loc':4,})

#绘制杭州边界
sz_all = gpd.GeoDataFrame()
sz_all['geometry'] = [sz.unary_union]
sz_all.plot(ax = ax,edgecolor = (0,0,0,1),facecolor = (0,0,0,0),linewidths=0.5)
#绘制标题
plt.axis('off')
plt.title('Hangzhou city taxi destination point distribution', fontsize=18, verticalalignment='center', horizontalalignment= 'center')
ax.set_xlim(118.0,121.0)
ax.set_ylim(29.0,31.0)
#绘制colorbar
plt.imshow([[0,vmax]], cmap=cmap)
cax = plt.axes([0.08, 0.4, 0.02, 0.3])
plt.colorbar(cax=cax)
##保存图片
plt.savefig('杭州市出租车下车点', dpi=1000, bbox_inches='tight')
plt.show()

#提取西湖区部分地区作为样本，减小数据量
lon1 = 120.0094224
lon2 = 120.1853465
lat1 = 30.0793895
lat2 = 30.3550171
df_ft = df[(df['Lng']>=lon1)&\
(df['Lng']<=lon2)&\
(df['Lat']>=lat1)&\
(df['Lat']<=lat2)]
data = df_ft
# 提取了17729347条数据

#把数据的每一列上移一行赋值给新的列。新的列名为原列名后面加‘1’
for c in data.columns:
    data[c+'1'] = data[c].shift(-1)
    print(c)

#生成轨迹ID
#删掉车辆id不同的行
data = data[(data['VehicleNum'].astype('int')==data['VehicleNum1'].fillna(0).astype('int'))]
#生成状态变化的列
data[['OpenStatus', 'OpenStatus1']] = data[['OpenStatus', 'OpenStatus1']].apply(pd.to_numeric)
data['StatusChange'] = data['OpenStatus1']-data['OpenStatus']
#为订单进行编号
data['neworder'] = (data['StatusChange'] == 1).astype('int')
data['orderid'] = data.groupby(['VehicleNum'])['neworder'].cumsum()
#整理数据的列名
data.columns=['VehicleNum', 'Stime', 'SLng', 'SLat', 'OpenStatus', 'Speed',
       'VehicleNum1', 'Etime', 'ELng', 'ELat', 'OpenStatus1', 'Speed1',
       'StatusChange', 'neworder', 'orderid']
data = data.drop(['Stime', 'SLng', 'SLat','Speed','OpenStatus','VehicleNum1','StatusChange', 'neworder' , 'OpenStatus1','Speed1'],axis = 1)
data.columns= ['driver_id', 'timestamp1', 'lon', 'lat','order_id']
df_xh =data

#将时间转为时间戳形式
#转为时间戳
df_xh['timestamp1'] = df_xh['timestamp1'].astype('datetime64')
from datetime import datetime
# review_date 转为时间戳形式
def time2stamp(cmnttime):   #转时间戳函数
    cmnttime=datetime.strptime(cmnttime,'%Y-%m-%d %H:%M:%S')
    stamp=int(datetime.timestamp(cmnttime))
    return stamp
df_xh['timestamp']=df_xh['timestamp1'].apply(lambda x: time2stamp(str(x)))
df_xh = df_xh.drop(['timestamp1'],axis = 1)

#将wgs84坐标系转换成投影坐标系
import pyproj
import pandas as pd
import numpy as np
p1 = pyproj.Proj(init="epsg:4326")#wgs坐标系统的EPSG Code
p2 = pyproj.Proj(init="epsg:3857")#球体墨卡托——转换投影坐标系统的EPSG Code
df_xh['x'],df_xh['y'] = pyproj.transform(p1, p2,df_xh['lon'], df_xh['lat'])
df_xh['x'] = round(df_xh['x'],2)#整列小数点保留2位
df_xh['y'] = round(df_xh['y'],2)#整列小数点保留2位
##保存数据
#df_xh.to_csv('Datatime.csv')
df_xh.fillna(0)##填补缺失值

#时间窗划分
stamp1 = 1642737032
stamp1 = int(stamp1)
time_interval=1200 #时间窗长度
df_xh['time_id'] = df_xh['timestamp'].apply(lambda x: (x - stamp1)//time_interval) #生成时间窗索引
#空间网格划分
left = df_xh['x'].min() #计算左边界
up = df_xh['y'].max() #计算上边界
interval=70 #网格单元大小

df_xh['rowid'] = df_xh['y'].apply(lambda x: (up - x) // interval).astype('int') #计算横向索引
df_xh['colid'] = df_xh['x'].apply(lambda x: (x - left) // interval).astype('int')#计算纵向索引

df_xh = df_xh.sort_values(by=['driver_id', 'order_id', 'timestamp']).reset_index(drop=True)
# 将订单id，下移一行，用于判断相邻记录是否属于同一订单
df_xh['orderFlag'] = df_xh['order_id'].shift(1)
df_xh['identi'] = (df_xh['orderFlag']==df_xh['order_id'])
# 将坐标、时间戳下移一行，从而匹配相邻轨迹点
df_xh['x1'] = df_xh['x'].shift(1)
df_xh['y1'] = df_xh['y'].shift(1)
df_xh['timestamp1'] = df_xh['timestamp'].shift(1)
df_xh = df_xh[df_xh['identi']==True]   #将不属于同一订单的轨迹点对删去
dist = np.sqrt(np.square((df_xh['x'].values-df_xh['x1'].values)) + np.square((df_xh['y'].values-df_xh['y1'].values)))    # 计算相邻轨迹点之间的距离
time = df_xh['timestamp'].values - df_xh['timestamp1'].values   # 计算相邻轨迹点相差时间
df_xh['speed'] = dist / time    # 计算速度
df_xh = df_xh.drop(columns=['x1', 'y1', 'orderFlag', 'timestamp1', 'identi'])   # 删去无用列

df_xh['speed1'] = df_xh.speed.shift(1)                 # 将速度下移一行
df_xh['timestamp1'] = df_xh.timestamp.shift(1)         # 将时间下移一行
df_xh['identi'] = df_xh.order_id.shift(1)              # 将订单号下移一行
df_xh = df_xh[df_xh.order_id==df_xh.identi]                  # 去除两个订单分界点数据
df_xh.loc[:, 'acc'] = (df_xh.speed1.values - df_xh.speed.values) / (df_xh.timestamp1.values - df_xh.timestamp.values)  #计算加速度
df_xh = df_xh.drop(columns=['speed1', 'timestamp1', 'identi'])  #删除临时字段

orderGrouped = df_xh.groupby(['rowid', 'colid','time_id', 'order_id'])  # 基于时空网格与轨迹id进行分组
# 网格平均车速
grouped_speed = orderGrouped.speed.mean().reset_index() # 重索引
grouped_speed = grouped_speed.groupby(['rowid', 'colid', 'time_id'])
grid_speed = grouped_speed.speed.mean()
grid_speed = grid_speed.clip(grid_speed.quantile(0.05), grid_speed.quantile(0.95))#去除异常值
grid_speed.head()

# 网格平均加速度
gridGrouped = df_xh.groupby(['rowid', 'colid','time_id'])
grid_acc = gridGrouped.acc.mean()
grid_acc.head()

# 网格流量
grouped_volume = orderGrouped.speed.last().reset_index()
grouped_volume = grouped_volume.groupby(['rowid', 'colid', 'time_id'])
grid_volume = grouped_volume['speed'].size()
grid_volume = grid_volume.clip(grid_volume.quantile(0.05), grid_volume.quantile(0.95))
grid_volume.head()

# 网格车速标准差
grid_v_std = gridGrouped.speed.std()
grid_v_std.head()

# 网格平均停车次数
stopNum = gridGrouped.speed.agg(lambda x: (x==0).sum())
grid_stop = pd.concat((stopNum, grid_volume), axis=1)
grid_stop['stopNum'] = stopNum.values / grid_volume.values
grid_stop = grid_stop['stopNum']
grid_stop = grid_stop.clip(0,grid_stop.quantile(0.95))
grid_stop.head()

# 合并网格数据
grib_data = pd.concat([grid_speed, grid_acc, grid_volume, grid_v_std, grid_stop], axis=1).reset_index()
grib_data.columns = ['rowid', 'colid', 'time_id', 'aveSpeed', 'gridAcc', 'volume', 'speed_std', 'stopNum']
grib_data.head()

#grib_data.to_csv('data_xhtaxi.csv')
print('数据预处理结束')
#grib_data = pd.read_csv('/Users/zipging/PycharmProjects/pythonProject4/data_xhtaxi.csv')