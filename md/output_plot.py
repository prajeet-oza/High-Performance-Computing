import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

N = 500

p1_500 = pd.read_csv('500_md_p1_6.csv')
p2_500 = pd.read_csv('500_md_p2_6.csv')
se_500 = pd.read_csv('500_md_se.csv')

plt.figure()
plt.plot(p1_500['Time'], p1_500['PE']/N, label = 'PE/N', linewidth = 1.5)
plt.plot(p1_500['Time'], p1_500['KE']/N, label = 'KE/N', linewidth = 1.5)
plt.plot(p1_500['Time'], p1_500['TE']/N, label = 'TE/N', linewidth = 1.5)
plt.plot(p1_500['Time'], p1_500['Temp'], label = 'Temperature', linewidth = 1.5)
plt.plot(p1_500['Time'], p1_500['Pres'], label = 'Pressure', linewidth = 1.5)
plt.title('Code Output, P1 app., 6 cores', fontsize=14)
plt.ylabel('Properties', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.legend()
plt.show()

plt.figure()
plt.plot(p2_500['Time'], p2_500['PE']/N, label = 'PE/N', linewidth = 1.5)
plt.plot(p2_500['Time'], p2_500['KE']/N, label = 'KE/N', linewidth = 1.5)
plt.plot(p2_500['Time'], p2_500['TE']/N, label = 'TE/N', linewidth = 1.5)
plt.plot(p2_500['Time'], p2_500['Temp'], label = 'Temperature', linewidth = 1.5)
plt.plot(p2_500['Time'], p2_500['Pres'], label = 'Pressure', linewidth = 1.5)
plt.title('Code Output, P2 app., 6 cores', fontsize=14)
plt.ylabel('Properties', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.legend()
plt.show()

plt.figure()
plt.plot(se_500['Time'], se_500['PE']/N, label = 'PE/N', linewidth = 1.5)
plt.plot(se_500['Time'], se_500['KE']/N, label = 'KE/N', linewidth = 1.5)
plt.plot(se_500['Time'], se_500['TE']/N, label = 'TE/N', linewidth = 1.5)
plt.plot(se_500['Time'], se_500['Temp'], label = 'Temperature', linewidth = 1.5)
plt.plot(se_500['Time'], se_500['Pres'], label = 'Pressure', linewidth = 1.5)
plt.title('Code Output, Serial app.', fontsize=14)
plt.ylabel('Properties', fontsize=14)
plt.xlabel('Steps', fontsize=14)
plt.legend()
plt.show()