import pandas as pd 
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="darkgrid", font_scale=1.5)
plt.style.use('ggplot')
# from sklearn.datasets import load_diabetes


# # 改变字体默认加粗
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

font = {'family':'Times New Roman', 'weight':'bold', 'size':12}
plt.rc('font', **font)


def average_smooth(x, smooth):
	'''
	x: 1d array
	'''
	smooth_x = []
	if smooth > 1:
		"""
		smooth data with moving window average.
		that is,
		    smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
		where the "smooth" param is width of that window (2k+1)
		"""
		y = np.ones(smooth)
		z = np.ones(len(x))
		smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
	return smoothed_x


root = '/home/xieliang/models/demos'

linewidth=1.5
shapes = ['triangle','pentagon','hexagon','diamond','trapezoid',
	'fillet-1','fillet-2','fillet-3','convex-1','convex-2','concave','cross']
eval_shapes = ['fillet-1', 'convex-1']


# figure = plt.figure(figsize=(10,8))
###################
ax = plt.subplot(2,2,1)
data1_1 = pd.read_csv(os.path.join(root, 'encoder/1_shape/encoder_distance_train.csv')).values[:,2]
sns.lineplot(data=data1_1, alpha=0.2, linewidth=1, color='steelblue')
g1 = sns.lineplot(data=average_smooth(data1_1,smooth=20), 
	linewidth=1, label='1-shape', color='steelblue')

data1_2 = pd.read_csv(os.path.join(root, 'encoder/3_shape/encoder_distance_train.csv')).values[:,2]
sns.lineplot(data=data1_2, alpha=0.2, linewidth=1, color='darkgray')
g1 = sns.lineplot(data=average_smooth(data1_2,smooth=20), 
	linewidth=1, label='3-shape', color='darkgray')

# data1_3 = pd.read_csv(os.path.join(root, 'encoder/5_shape/encoder_distance_train.csv')).values[:,2]
# sns.lineplot(data=data1_3, alpha=0.2, linewidth=linewidth, color='red')
# g1 = sns.lineplot(data=average_smooth(data1_3,smooth=20), 
# 	linewidth=linewidth, label='5-shape', color='red')

# g1.legend(prop={'weight':'light', 'size':11})
g1.legend(prop={'weight':'light', 'size':11}, title='From scratch(17h)', title_fontsize='11', fontsize=11)
ax.set_xlabel(None)
# ax.set_ylabel('Distance Error', fontdict=font)
plt.ylabel('Distance Error',  fontweight='bold', fontsize='13')
ax.set_yticks([0.0,2.5,5.0,7.5,10.0])
# ax.set_title("From scratch (17h)", fontsize='13', fontweight='bold')


# ###################
ax = plt.subplot(2,2,2)

df3 = pd.DataFrame(columns=['index']+eval_shapes)
df3['index'] = range(10)
for shape_index, shape in enumerate(eval_shapes):
	data1 = pd.read_csv(os.path.join(root, 'eval/adaption_ori/encoder/1_shape', shape, 'distance_train.csv'))
	df3[shape] = data1['Value'].values[0:780:78]
	ax = sns.lineplot(data=df3, x='index', y=shape, linewidth=linewidth, 
		label=shape)
	ax.lines[shape_index].set_linestyle('dashdot')

df2 = []
for shape_index, shape in enumerate(shapes):
	data1 = pd.read_csv(os.path.join(root, 'eval/adaption_ori/encoder/1_shape', shape, 'distance_train.csv'))
	data1_per_epoch = data1[0:780:78]
	data1_per_epoch['index']=range(10)
	data1_per_epoch['shape_type'] = [shape_index]*10
	df2.append(data1_per_epoch.values.tolist())
df2 = pd.DataFrame(data=np.array(df2).reshape(-1,5), columns=['Wall time', 'Step', 'Value', 'index', 'shape_type'])
g2 = sns.lineplot(data=df2, x='index', y='Value', ci=100, 
	linewidth=linewidth, label='1-shape-avg', marker='o', 
	markersize=5, alpha=0.5)


df4 = []
shapes_1 = ['diamond','fillet-1','fillet-2','fillet-3','convex-1','convex-2','concave','cross']
for shape_index, shape in enumerate(shapes_1):
	data1 = pd.read_csv(os.path.join(root, 'eval/adaption_ori/encoder/3_shape', shape, 'distance_test.csv'))
	data1_per_epoch = data1[0:780:78]
	data1_per_epoch['index']=range(10)
	data1_per_epoch['shape_type'] = [shape_index]*10
	df4.append(data1_per_epoch.values.tolist())
df4 = pd.DataFrame(data=np.array(df4).reshape(-1,5), columns=['Wall time', 'Step', 'Value', 'index', 'shape_type'])
g2 = sns.lineplot(data=df4, x='index', y='Value', ci=100, 
	linewidth=linewidth, label='3-shape-avg', marker='o', 
	markersize=5, alpha=0.5)

g2.set(xlabel=None)
g2.set(ylabel=None)
g2.legend(prop={'weight':'light', 'size':11}, title='Adaption(3min)', title_fontsize='11')
# g2.legend(prop={'weight':'light', 'size':11})
ax.set_yticks([0.0,2.5,5.0,7.5,10.0])





# ###################
ax = plt.subplot(2,2,3)
data3_1 = pd.read_csv(os.path.join(root, 'rl/1_shape/RL_TrainAvgEpRet.csv')).values[:,2]
sns.lineplot(data=average_smooth(data3_1, smooth=5), 
	alpha=0.2, linewidth=linewidth, color='steelblue')
g3 = sns.lineplot(data=average_smooth(data3_1, smooth=45), 
	linewidth=linewidth, color='steelblue',
	label='1-shape')

data3_2 = pd.read_csv(os.path.join(root, 'rl/3_shape/RL_TrainAvgEpRet.csv')).values[:,2]
sns.lineplot(data=average_smooth(data3_2, smooth=5), 
	alpha=0.2, linewidth=linewidth, color='darkgray')
g3 = sns.lineplot(data=average_smooth(data3_2, smooth=45), 
	linewidth=linewidth, color='darkgray',
	label='3-shape')


# data3_3 = pd.read_csv(os.path.join(root, 'rl/5_shape/RL_TrainAvgEpRet.csv')).values[:,2]
# sns.lineplot(data=average_smooth(data3_3, smooth=5), 
# 	alpha=0.2, linewidth=linewidth, color='green')
# g3 = sns.lineplot(data=average_smooth(data3_3, smooth=45), 
# 	linewidth=linewidth, color='green',
# 	label='5-shape')


g3.legend(prop={'weight':'light', 'size':11}, title='From scratch(4h)', title_fontsize='11', fontsize=11)
plt.xlabel('Epoch', fontweight='bold', fontsize='13')
plt.ylabel('Success Rate', fontweight='bold', fontsize='13')
# ax.set_xlabel('Epoch', fontdict=font)
# ax.set_ylabel('Success', fontdict=font)
ax.set_yticks([0.0,0.2,0.4,0.6, 0.8,1.0])




# #####################################################
ax = plt.subplot(2,2,4)

df5 = pd.DataFrame(columns=['index']+eval_shapes)
df5['index'] = range(3)
for shape_index, shape in enumerate(eval_shapes):
	data1 = pd.read_csv(os.path.join(root, 'eval/adaption_ori/rl/1_shape/', shape, 'EvalAvgSuccess.csv'))
	print(shape)
	print(data1)
	data1['Value'] = average_smooth(data1.values[:,2], smooth=20)
	print(data1)
	df5[shape] = data1['Value'].values[5:78*3+5:78]
	# print(df5[shape])
	ax = sns.lineplot(data=df5, x='index', y=shape, linewidth=linewidth, 
		label=shape, palette=sns.color_palette('deep'))
	ax.lines[shape_index].set_linestyle('dashdot')


df4 = []
for shape in shapes:
	data4 = pd.read_csv(os.path.join(root, 'eval/adaption_ori/rl/1_shape', shape, 'EvalAvgSuccess.csv'))
	data4['Value'] = average_smooth(data4.values[:,2], smooth=3)
	data4_per_epoch = data4[0:78*3:78]
	data4_per_epoch['index']=range(3)
	data4_per_epoch['type'] = [0]*3
	df4.append(data4_per_epoch.values.tolist())
df4 = pd.DataFrame(data=np.array(df4).reshape(-1,5), columns=['Wall time', 'Step', 'Value', 'index', 'type'])
g4 = sns.lineplot(data=df4, x='index', y='Value', ci=90, linewidth=linewidth, 
	label='1-shape-avg', marker='o', markersize=5, alpha=0.5,
	palette=sns.color_palette('deep'))


df5 = []
for shape in shapes_1:
	data4 = pd.read_csv(os.path.join(root, 'eval/adaption_ori/rl/3_shape', shape, 'EvalAvgSuccess.csv'))
	data4['Value'] = average_smooth(data4.values[:,2], smooth=6)
	data4_per_epoch = data4[2:300*3:300]
	data4_per_epoch['index']=range(3)
	data4_per_epoch['type'] = [0]*3
	df5.append(data4_per_epoch.values.tolist())
df5 = pd.DataFrame(data=np.array(df5).reshape(-1,5), columns=['Wall time', 'Step', 'Value', 'index', 'type'])
g4 = sns.lineplot(data=df5, x='index', y='Value', ci=90, linewidth=linewidth, 
	label='3-shape-avg', marker='o', markersize=5, alpha=0.5,
	palette=sns.color_palette('deep'))


# g4.set(xlabel='Epoch')
g4.set(ylabel=None)
# ax.set_xlabel('Epoch', fontdict=font)
plt.xlabel('Epoch',  fontweight='bold', fontsize='13')
# ax.set_xticks([0,1,2])
# ax.set_yticks([0.75,0.8,0.85,0.9,0.95,1.0])
ax.set_yticks([0.7,0.8,0.9,1.0])
# ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
g4.legend(prop={'weight':'light','size':11}, title='Adaption(1min)', title_fontsize='11', loc='lower right')



########################
# plt.xlabel('Epoch')
# plt.legend(prop={'weight':'light'}, title='Adaption(1min)', title_fontsize='11', fontsiz=11, loc='lower right')
plt.show()



