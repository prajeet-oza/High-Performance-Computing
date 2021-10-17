# ABOUT THE CODE:
# The code integrates cosine over the limits, A and B,
# with parallel processing the integration in parts at each process,
# where each process serially calculates the integral.

# importing relevant libraries
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

# take input for rank and size
rank = comm.Get_rank()
size = comm.Get_size()

n = 500 # number of split of the integration limits for each process
a = 0
b = np.pi / 2 # limits of the integration

h = (b - a) / (size * n) # increment in x during the integration

intg = np.zeros(1) # initialise a variable for the total value for integration after summing all the processes
intgi = np.zeros(1) # initialise a variable for the value of integration for a process
data = np.zeros(1) # send-receive variable

# serial integration of the cosine function for the process
ai = a + rank * n * h 
for j in range(n):
	aij = ai + h * (j + 0.5)
	intgi[0] += np.cos(aij) * h
print('Process', rank, 'has integral value :', intgi[0]) # output of integral value for each process


# comm.Reduce(intgi, intg, op = MPI.SUM, root = 0) # easy way is to use reduce method with sum operation
# other way is to use send and receive methods to sum the integral values obtained from each process
if rank != 0:
	comm.Send(intgi, dest = 0)

if rank == 0:
	intg[0] = intgi[0]
	for i in range(1, size):
		comm.Recv(data, source = i)
		intg[0] += data[0]
	print('cos integral over', a, 'and', b, ':', intg[0]) # total intregral value for the cosine function