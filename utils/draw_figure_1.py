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


plt.subplot(2,2,1)
#################################
rl_seen_10_step = np.array([0.29])
trans_naive_seen_10_step = np.array([0.89])
trans_rl_seen_10_step = np.array([0.96])
e2e_trans_rl_seen_10_step = np.array([0.99])

df1 = pd.DataFrame(columns=['Method', 'Success'])
df1['Method'] = ['RL']*1
df1['Success'] = rl_seen_10_step

df2 = pd.DataFrame(columns=['Method', 'Success'])
df2['Method'] = ['Trans+Naive']*1
df2['Success'] = trans_naive_seen_10_step

df3 = pd.DataFrame(columns=['Method', 'Success'])
df3['Method'] = ['Fixed_Trans+RL']*1
df3['Success'] = trans_rl_seen_10_step

df4 = pd.DataFrame(columns=['Method', 'Success'])
df4['Method'] = ['E2E Trans+RL']*1
df4['Success'] = e2e_trans_rl_seen_10_step

df = pd.concat([df1,df2,df3,df4], ignore_index=True)
ax = sns.barplot(data=df, x='Method', y='Success')
ax.bar_label(ax.containers[0], padding=-20, fontweight='light', fontsize='11')
ax.set_title("Seen shape", fontsize=13, fontweight='bold')
ax.set(xlabel=None)
plt.ylabel('10-step', fontsize=13, fontweight='bold')
plt.xticks([])



plt.subplot(2,2,2)
###############################
rl_unseen_10_step = np.array([0.32,0.28,0.28,0.30,0.29,0.26,0.28,0.27,0.31,0.28,0.30,0.26])
trans_naive_unseen_10_step = np.array([0.67,0.70,0.71,0.66,0.71,0.54,0.81,0.60,0.77,0.68,0.61,0.80])
trans_rl_unseen_10_step = np.array([0.66,0.58,0.66,0.60,0.68,0.60,0.80,0.61,0.91,0.53,0.60,0.88])
e2e_trans_rl_unseen_10_step = np.array([0.75,0.77,0.73,0.72,0.84,0.81,0.93,0.74,0.94,0.82,0.83,0.92])

df1 = pd.DataFrame(columns=['Method', 'Success'])
df1['Method'] = ['RL']*12
df1['Success'] = rl_unseen_10_step

df2 = pd.DataFrame(columns=['Method', 'Success'])
df2['Method'] = ['Trans+Naive']*12
df2['Success'] = trans_naive_unseen_10_step

df3 = pd.DataFrame(columns=['Method', 'Success'])
df3['Method'] = ['Fixed_Trans+RL']*12
df3['Success'] = trans_rl_unseen_10_step

df4 = pd.DataFrame(columns=['Method', 'Success'])
df4['Method'] = ['E2E Trans+RL']*12
df4['Success'] = e2e_trans_rl_unseen_10_step

df = pd.concat([df1,df2,df3,df4], ignore_index=True)
ax = sns.barplot(data=df, x='Method', y='Success')
labels = [0.28, 0.68, 0.68, 0.82]
ax.bar_label(ax.containers[0], padding=-20, labels=labels, fontweight='light', fontsize='11')
ax.set(xlabel=None)
ax.set(ylabel=None)
ax.set_title("Unseen shapes", fontweight='bold', fontsize=13)
# ax.set(ylabel='Success', fontweight='bold')
plt.xticks([])


plt.subplot(2,2,3)
#################################
rl_seen_20_step = np.array([0.34])
trans_naive_seen_20_step = np.array([0.89])
trans_rl_seen_20_step = np.array([0.97])
e2e_trans_rl_seen_20_step = np.array([0.99])

df1 = pd.DataFrame(columns=['Method', 'Success'])
df1['Method'] = ['RL']*1
df1['Success'] = rl_seen_20_step

df2 = pd.DataFrame(columns=['Method', 'Success'])
df2['Method'] = ['Trans+Naive']*1
df2['Success'] = trans_naive_seen_20_step

df3 = pd.DataFrame(columns=['Method', 'Success'])
df3['Method'] = ['Fixed_Trans+RL']*1
df3['Success'] = trans_rl_seen_20_step

df4 = pd.DataFrame(columns=['Method', 'Success'])
df4['Method'] = ['E2E Trans+RL']*1
df4['Success'] = e2e_trans_rl_seen_20_step

df = pd.concat([df1,df2,df3,df4], ignore_index=True)
ax = sns.barplot(data=df, x='Method', y='Success')
ax.bar_label(ax.containers[0], padding=-20, fontweight='light', fontsize='11')
ax.set(xlabel=None)
plt.ylabel('20-step', fontweight='bold', fontsize='13')
plt.xticks([])





plt.subplot(2,2,4)
#####################################
rl_unseen_20_step = np.array([0.31,0.36,0.37,0.35,0.33,0.31,0.32,0.34,0.37,0.36,0.34,0.36])
trans_naive_unseen_20_step = np.array([0.68,0.68,0.74,0.70,0.64,0.58,0.78,0.69,0.74,0.61,0.66,0.85])
trans_rl_unseen_20_step = np.array([0.76,0.76,0.90,0.78,0.86,0.69,0.92,0.78,0.91,0.58,0.74,0.96])
e2e_trans_rl_unseen_20_step = np.array([0.92,0.85,0.88,0.85,0.92,0.88,0.90,0.86,0.97,0.88,0.96,0.98])

df1 = pd.DataFrame(columns=['Method', 'Success'])
df1['Method'] = ['RL']*12
df1['Success'] = rl_unseen_20_step

df2 = pd.DataFrame(columns=['Method', 'Success'])
df2['Method'] = ['Trans+Naive']*12
df2['Success'] = trans_naive_unseen_20_step

df3 = pd.DataFrame(columns=['Method', 'Success'])
df3['Method'] = ['Fixed_Trans+RL']*12
df3['Success'] = trans_rl_unseen_20_step

df4 = pd.DataFrame(columns=['Method', 'Success'])
df4['Method'] = ['E2E Trans+RL']*12
df4['Success'] = e2e_trans_rl_unseen_20_step

df = pd.concat([df1,df2,df3,df4], ignore_index=True)
ax = sns.barplot(data=df, x='Method', y='Success')
labels = ['0.34', '0.68', '0.79', '0.90']
ax.bar_label(ax.containers[0], padding=-20, labels=labels, fontweight='light', fontsize='11')
ax.set(xlabel=None)
ax.set(ylabel=None)
plt.xticks([])

#################################
# plt.legend(loc='upper center', labels=['RL', 'Trans+Naive', 'Fixed_Trans+RL', 'E2E Trans+RL'])
plt.show()
exit(0)




