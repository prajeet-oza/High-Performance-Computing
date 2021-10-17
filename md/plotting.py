import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

p1_data = pd.read_csv('p1_data.csv', index_col = 'procs')
p1_cols = p1_data.columns.values
p1_ids = p1_data.index.values

p2_data = pd.read_csv('p2_data.csv', index_col = 'procs')
p2_cols = p2_data.columns.values
p2_ids = p2_data.index.values

plt.rcParams.update({'font.family':'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12})
fig = plt.figure()
ax = fig.add_subplot(111)
axi = np.zeros(6, dtype = object)
plt.suptitle('Relative speedup vs Number of process, both decomp.', fontsize = 24)
for i in range(1, 3):
	for j in range(1, 4):
		id_ = 3*i + j - 3
		sp = '23' + str(id_)
		if id_ != 6:
			col = p1_cols[id_-1]
			axi[id_-1] = fig.add_subplot(int(sp))
			axi[id_-1].plot(p1_ids, p1_data.loc[:, col].values[0]/p1_data.loc[:, col].values, label = col+'_p1', linewidth = 3)
			axi[id_-1].plot(p1_ids, p2_data.loc[:, col].values[0]/p2_data.loc[:, col].values, label = col+'_p2', linewidth = 3)
			axi[id_-1].set_ylim([0, 2.5])
			axi[id_-1].set_title('No. atoms: {}'.format(col), fontsize = 16, pad = 2)
			axi[id_-1].legend(loc = 'upper left', fontsize = 10)
axi[-1] = fig.add_subplot(236)
for col in ['50', '1000']:
	axi[-1].plot(p1_ids, p1_data.loc[:, col].values[0]/p1_data.loc[:, col].values, label = col+'_p1', linewidth = 3)
	axi[-1].plot(p1_ids, p2_data.loc[:, col].values[0]/p2_data.loc[:, col].values, label = col+'_p2', linewidth = 3)
axi[-1].set_ylim([0, 2.5])
axi[-1].set_title('Comparing the speedup', fontsize = 16, pad = 2)
axi[-1].legend(loc = 'upper left', fontsize = 10)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
ax.set_xlabel('Number of processors', fontsize = 22)
ax.set_ylabel('Relative speedup', fontsize = 22)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
axi = np.zeros(9, dtype = object)
plt.suptitle('Time per atom vs Number of atoms, both decomp.', fontsize = 24)
for i in range(1, 4):
	for j in range(1, 4):
		id_ = 3*i + j - 3
		sp = '33' + str(id_)
		if id_ != 9:
			axi[id_-1] = fig.add_subplot(int(sp))
			axi[id_-1].plot(p1_cols, p1_data.loc[id_].values/p1_cols.astype('int'), label = str(id_)+'_p1', linewidth = 2)
			axi[id_-1].plot(p1_cols, p2_data.loc[id_].values/p2_cols.astype('int'), label = str(id_)+'_p2', linewidth = 2)
			axi[id_-1].set_ylim([0, 2])
			axi[id_-1].set_title('Process {}'.format(id_), fontsize = 16, pad = 2)
			axi[id_-1].legend(loc = 'upper left', fontsize = 10)
axi[-1] = fig.add_subplot(339)
for id_ in [1, 2, 6, 8]:
	axi[-1].plot(p1_cols, p1_data.loc[id_].values/p1_cols.astype('int'), label = str(id_)+'_p1', linewidth = 2)
	axi[-1].plot(p1_cols, p2_data.loc[id_].values/p2_cols.astype('int'), label = str(id_)+'_p2', linewidth = 2)
axi[-1].set_ylim([0, 2])
axi[-1].set_title('Comparing the processors', fontsize = 16, pad = 2)
axi[-1].legend(loc = 'upper left', fontsize = 10)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
ax.set_xlabel('Number of atoms', fontsize = 22)
ax.set_ylabel('Time per atom', fontsize = 22)
plt.show()
