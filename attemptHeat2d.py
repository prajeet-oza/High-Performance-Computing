# importing relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def initialCon(x_val, y_val): # initial condition function, so accordingly change the function
	initCon = np.array([])
	for i in range(x_val.shape[0]):
		initCon = np.append(initCon, x_val[i] * y_val)
	initCon = initCon.reshape((x_val.shape[0], y_val.shape[0]))
	return initCon.T

comm = MPI.COMM_WORLD

# getting input for rank and size
size = comm.Get_size()
rank = comm.Get_rank()

time_parallel = np.zeros(1) # variable storing time for each process
total_time = np.zeros(1) # total time after adding time from each process

start_parallel = MPI.Wtime() # starting solving of PDE

# specifying the conditions for the problem,
delX = 0.1
delY = 0.1
gamma = 2 * (delX**2 + delY**2) / (delX * delY)**2

lengthX, lengthY = 1, 1 # length of rod

propB, propR = 0, 0 # boundary conditions
propU = 100

error = np.inf
tolerance = 1 / 10**4

# NOTATION: J refers to X, and N refers to time
J_root = int(lengthX / delX) # splitting the rod into delX pieces
K_root = int(lengthY / delY)

J_local = J_root // size # splitting the delX pieces into domains
K_local = K_root
if rank == size - 1:
	J_local = J_root // size + J_root % size + 1
# the last process will account for the residual pieces once they are evenly distributed among all process

# NOTATION: prop refers to the property, here it is temperature
prop_root = np.zeros((K_root + 1, J_root + 1)) # variable used to finally gather all PDE solutions from process
prop_local = np.zeros((K_local + 1, J_local)) # local variable used to solve for PDE solutions for the domain

minX_local = rank * (J_root // size) * delX # min value for the domain
maxX_local = minX_local + (J_local - 1) * delX # max value for the domain

minY_local = 0
maxY_local = lengthY

# setting up the initial condition
x_local = np.linspace(minX_local, maxX_local, J_local)
y_local = np.linspace(minY_local, maxY_local, K_local + 1)
propInit_local = initialCon(x_local, y_local)
prop_local = propInit_local.copy() # storing the initial conditions to local variable

# setting up the initial condition for the root variable too
if rank == 0:
	x = np.linspace(0, lengthX, J_root + 1)
	y = np.linspace(0, lengthY, K_root + 1)
	propInit = initialCon(x, y)
	prop_root = propInit.copy()
	prop_root[:, -1] = propR
	prop_root[-1] = propB
	prop_root[0] = propU
	copy_root = prop_root.copy()

if rank == size - 1:
	prop_local[:, -1] = propR # setting the boundary condition

prop_local[-1] = propB # setting the boundary condition
prop_local[0] = propU # setting the boundary condition

# defining a temporary variable for the tridiagonal matrix
K_temp = K_local - 2
if rank == size - 1:
	J_temp = J_local - 1
else:
	J_temp = J_local

# defining the tridiagonal matrix
qX = np.ones(J_temp - 1)
triMatX = np.diag(qX, k = -1) + np.diag(qX, k = 1)
# print(prop_local[3, :].shape, flush = True)

qY = gamma * np.ones(K_temp - 1)
triMatY = np.diag(qY, k = -1) + np.diag(qY, k = 1)

# carrying out the explicit method to solve the PDE
count = 0
max_iter = 3
prop_local_old = prop_local.copy()
# while error > tolerance:
while count < max_iter:
	for yi in range(K_root - 1):
		tempX = np.zeros(J_temp) # temporary variable to account for the boundaries
		tempY = np.zeros(K_temp) # temporary variable to account for the boundaries
		tempY[0] = propU
		tempY[-1] = propB
		# send and receive variable to communicate between left and right processes
		sendL = np.zeros(1)
		sendR = np.zeros(1)
		recvL = np.zeros(1)
		recvR = np.zeros(1)

		if rank == 0: # rank 0 process condition, dealing with left boundary condition
			comm.Recv(recvR, source = rank + 1, tag = 10)
			tempX[0] = prop_local_old[yi, 1]
			tempX[-1] = recvR[0]

			mulMat = np.dot(triMatX, prop_local_old[yi, :].T)
			prop_local[yi, :] = ((mulMat + tempX.T) / delX**2 + (prop_local_old[yi+1, :] + prop_local[yi-1, :]) / delY**2) / gamma

			sendR[0] = prop_local_old[yi, -1]
			comm.Send(sendR, dest = rank + 1, tag = 20)

		elif rank == size - 1: # rank (size - 1) process condition, dealing with right boundary condition
			sendL[0] = prop_local_old[yi, 0]
			comm.Send(sendL, dest = rank - 1, tag = 10)
			comm.Recv(recvL, source = rank - 1, tag = 20)
			tempX[0] = recvL[0]

			mulMat = np.dot(triMatX, prop_local_old[yi, :-1].T)
			prop_local[yi, :-1] = ((mulMat + tempX.T) / delX**2 + (prop_local_old[yi+1, :-1] + prop_local[yi-1, :-1]) / delY**2) / gamma
			# tempX[-1] = prop_local[yi, -1]

		else: # other process condition, dealing with center portions of the rod
			sendL[0] = prop_local_old[yi, 0]
			comm.Send(sendL, dest = rank - 1, tag = 10) # sending to the right process
			comm.Recv(recvR, source = rank + 1, tag = 10) # receiving from the right process
			comm.Recv(recvL, source = rank - 1, tag = 20) # receiving from the left process
			tempX[0] = recvL[0]
			tempX[-1] = recvR[0]

			mulMat = np.dot(triMatX, prop_local_old[yi, :].T)
			prop_local[yi, :] = ((mulMat + tempX.T) / delX**2 + (prop_local_old[yi+1, :] + prop_local[yi-1, :]) / delY**2) / gamma

			sendR[0] = prop_local_old[yi, -1]
			comm.Send(sendR, dest = rank + 1, tag = 20) # sending to the left process

	prop_local_old = prop_local.copy()
	count += 1

if rank == 0: # gathering all the data from the process to compile the PDE
	prop_root[:, rank * J_local : (rank+1) * J_local] = prop_local.copy()
	for proc in range(1, size):
		if proc == size - 1:
			recvArray = np.zeros((K_root + 1, J_local + J_root % size + 1))
			comm.Recv(recvArray, source = proc)
			prop_root[:, proc * J_local:] = recvArray.copy()
		else:
			recvArray = np.zeros((K_root + 1, J_local))
			comm.Recv(recvArray, source = proc)
			prop_root[:, proc * J_local : (proc+1) * J_local] = recvArray.copy()
else:
	comm.Send(prop_local, dest = 0)

stop_parallel = MPI.Wtime() # stop solving of PDE

time_parallel[0] = stop_parallel - start_parallel
comm.Reduce(time_parallel, total_time, op = MPI.SUM, root = 0)

if rank == 0: # get output for total time elapsed during solving of PDE
	print('\nTotal time for parallel PDE solving: ', total_time[0])

# plotting the solution to the PDE
if rank == 0:
	meshX, meshT = np.meshgrid(np.linspace(0, lengthX, J_root + 1), np.linspace(0, lengthY, K_root + 1))
	plt.figure()
	plt.contourf(meshX, meshT, np.rot90(prop_root, k = 2), 50, cmap='jet')
	plt.colorbar()
	plt.title('Solving 2D PDE Heat Equation', fontsize=13)
	plt.xlabel('x, rod length', fontsize=12)
	plt.ylabel('t, time', fontsize=12)
	plt.savefig('attempt_2d.png')
	# plt.show()
