import pandas as pd
import matplotlib.pyplot as plt

file1 = open('log.lammps', 'r')
lines = file1.readlines()
start = 71
length = 1000
data_lmp = lines[start:start+length+2]
file1.close()

file2 = open('lammps_data', 'w')
file2.writelines(data_lmp)
file2.close()

data_ = pd.read_table('lammps_data', sep = '\s+')
data_500 = data_[data_['Step'] % 10 == 0]

plt.figure()
plt.plot(data_500['Step'], data_500['PotEng'], label = 'PE/N', linewidth = 1.5)
plt.plot(data_500['Step'], data_500['KinEng'], label = 'KE/N', linewidth = 1.5)
plt.plot(data_500['Step'], data_500['TotEng'], label = 'TE/N', linewidth = 1.5)
plt.plot(data_500['Step'], data_500['Temp'], label = 'Temperature', linewidth = 1.5)
plt.plot(data_500['Step'], data_500['Press'], label = 'Pressure', linewidth = 1.5)
plt.title('LAMMPS Output', fontsize=14)
plt.ylabel('Properties', fontsize=14)
plt.xlabel('Steps', fontsize=14)
plt.legend()
plt.show()

# print('Enter the required number of plots: ')
# N = int(input())

# plt.figure()
# print('Enter x axis: ')
# x_int = int(input())
# x = header[x_int-1]
# for i in range(N):
#     print('Enter yi axis: ')
#     y_int = int(input())
#     y = header[y_int-1]
#     plt.plot(data_[x], data_[y], label = y)
# plt.legend()
# plt.show()