# This problem reaches equilibrium between 1000 seconds and 5000 seconds.

# importing relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def initialCon(x_val): # initial condition function, say the initial condition was 4x(1-x), so accordingly change the function
	initCon = 0 * x_val
	return initCon

comm = MPI.COMM_WORLD

# getting input for rank and size
size = comm.Get_size()
rank = comm.Get_rank()

time_parallel = np.zeros(1) # variable storing time for each process
total_time = np.zeros(1) # total time after adding time from each process

start_parallel = MPI.Wtime() # starting solving of PDE

# specifying the conditions for the problem,
delX = 0.01
delT = 0.1
alpha = 1 / 10**4
gamma = alpha * delT / delX**2

length = 1 # length of rod
Nsteps = 50000 # number of steps, so time = steps times delT
time = Nsteps * delT

propL, propR = 0, 100 # boundary conditions

# NOTATION: J refers to X, and N refers to time
J_root = int(length / delX) # splitting the rod into delX pieces

J_local = J_root // size # splitting the delX pieces into domains
if rank == size - 1:
	J_local = J_root // size + J_root % size + 1
# the last process will account for the residual pieces once they are evenly distributed among all process
N_root = int(time / delT) # splitting the time into delT pieces

# NOTATION: prop refers to the property, here it is temperature
prop_root = np.zeros((N_root + 1, J_root + 1)) # variable used to finally gather all PDE solutions from process
prop_local = np.zeros((N_root, J_local)) # local variable used to solve for PDE solutions for the domain
minX_local = rank * (J_root // size) * delX # min value for the domain
maxX_local = minX_local + (J_local - 1) * delX # max value for the domain
# setting up the initial condition
x_local = np.linspace(minX_local, maxX_local, J_local)
print(x_local, flush = True)
propInit_local = initialCon(x_local)
prop_local[0] = propInit_local.copy() # storing the initial conditions to local variable

# setting up the initial condition for the root variable too
if rank == 0:
	x = np.arange(0, length + delX, delX)
	propInit = initialCon(x)
	prop_root[0] = propInit.copy()
	prop_root[0, 0] = propL
	prop_root[0, -1] = propR
	prop_local[:, 0] = propL # setting the boundary condition
elif rank == size - 1:
	prop_local[:, -1] = propR # setting the boundary condition

# defining a temporary variable for the tridiagonal matrix
if rank == 0 or rank == size - 1:
	J_temp = J_local - 1
else:
	J_temp = J_local

# defining the tridiagonal matrix
p = (1 - 2 * gamma) * np.ones(J_temp)
q = gamma * np.ones(J_temp - 1)
triMat = np.diag(p) + np.diag(q, k = -1) + np.diag(q, k = 1)

# carrying out the explicit method to solve the PDE
for t in range(N_root - 1):
	temp = np.zeros(J_temp) # temporary variable to account for the boundaries
	# send and receive variable to communicate between left and right processes
	sendL = np.zeros(1)
	sendR = np.zeros(1)
	recvL = np.zeros(1)
	recvR = np.zeros(1)

	if rank == 0: # rank 0 process condition, dealing with left boundary condition
		sendR[0] = prop_local[t, -1]
		comm.Send(sendR, dest = rank + 1, tag = 20)
		comm.Recv(recvR, source = rank + 1, tag = 10)
		temp[0] = prop_local[t, 0]
		temp[-1] = recvR[0]
	elif rank == size - 1: # rank (size - 1) process condition, dealing with right boundary condition
		sendL[0] = prop_local[t, 0]
		comm.Send(sendL, dest = rank - 1, tag = 10)
		comm.Recv(recvL, source = rank - 1, tag = 20)
		temp[0] = recvL[0]
		temp[-1] = prop_local[t, -1]
	else: # other process condition, dealing with center portions of the rod
		sendL[0] = prop_local[t, 0]
		sendR[0] = prop_local[t, -1]
		comm.Send(sendL, dest = rank - 1, tag = 10) # sending to the right process
		comm.Send(sendR, dest = rank + 1, tag = 20) # sending to the left process
		comm.Recv(recvR, source = rank + 1, tag = 10) # receiving from the right process
		comm.Recv(recvL, source = rank - 1, tag = 20) # receiving from the left process
		temp[0] = recvL[0]
		temp[-1] = recvR[0]
	temp = gamma * temp.copy()

	if rank == 0: # last step in solving PDE, multiplying with the tridiagonal matrix
		mulMat = np.dot(triMat, prop_local[t, 1:].T)
		prop_local[t+1, 1:] = mulMat + temp.T
	elif rank == size - 1:
		mulMat = np.dot(triMat, prop_local[t, :-1].T)
		prop_local[t+1, :-1] = mulMat + temp.T
	else:
		mulMat = np.dot(triMat, prop_local[t, :].T)
		prop_local[t+1, :] = mulMat + temp.T

if rank == 0: # gathering all the data from the process to compile the PDE
	prop_root[1:, rank * J_local : (rank+1) * J_local] = prop_local.copy()
	for proc in range(1, size):
		if proc == size - 1:
			recvArray = np.zeros((N_root, J_local + J_root % size + 1))
			comm.Recv(recvArray, source = proc)
			prop_root[1:, proc * J_local:] = recvArray.copy()
		else:
			recvArray = np.zeros((N_root, J_local))
			comm.Recv(recvArray, source = proc)
			prop_root[1:, proc * J_local : (proc+1) * J_local] = recvArray.copy()
else:
	comm.Send(prop_local, dest = 0)

stop_parallel = MPI.Wtime() # stop solving of PDE

time_parallel[0] = stop_parallel - start_parallel
comm.Reduce(time_parallel, total_time, op = MPI.SUM, root = 0)

if rank == 0: # get output for total time elapsed during solving of PDE
	print('\nTotal time for parallel PDE solving: ', total_time[0])
	print(prop_root.shape)

# plotting the solution to the PDE
if rank == 0:
	meshX, meshT = np.meshgrid(np.arange(0, length + delX, delX), np.arange(0, time + delT, delT))
	print(meshX.shape, meshT.shape)
	plt.figure()
	plt.contourf(meshX, meshT, prop_root, 50, cmap='jet')
	plt.colorbar()
	plt.title('Solving 1D PDE Heat Equation', fontsize=13)
	plt.xlabel('x, rod length', fontsize=12)
	plt.ylabel('t, time', fontsize=12)
	plt.savefig(str(time)+'-sec-solution_test_3.png')
	# plt.show()
