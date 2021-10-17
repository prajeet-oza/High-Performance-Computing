# Btw, this code works with any number of processors? Atleast my limited testing showed it working with 3, 7, 9 etc.

# relevant libraries
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD

# getting rank and size input
rank = comm.Get_rank()
size = comm.Get_size()

time_parallel = np.zeros(1) # variable storing time for each process
total_time = np.zeros(1) # total time after adding time from each process

start_parallel = MPI.Wtime() # starting solving of PDE

if rank == 0:
	print('Starting the PDE solution, ...\n', flush = True)

# system information
lenX = 1
lenY = 1
delX = 0.01
delY = 0.01 

# number of bits for the x and y directions
nX = int(lenX / delX + 1)
nY = int(lenY / delY + 1)

# useful constant, terming it as gamma
gamma = 2 * (delX**2 + delY**2) / (delX * delY)**2

def triMatAlgo(a, b, c, d):	# tridiagonal matrix algorithm
	# references: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
	n = len(d)
	cp = np.zeros(n-1)
	dp = np.zeros(n)
	x = np.zeros(n)

	cp[0] = c[0] / b[0]
	for i in range(1, n-1):
		cp[i] = c[i] / (b[i] - a[i-1] * cp[i-1])

	dp[0] = d[0] / b[0]
	for i in range(1, n):
		dp[i] = (d[i] - a[i-1] * dp[i-1]) / (b[i] - a[i-1] * cp[i-1])
	
	x[n-1] = dp[n-1]
	for i in range(n-1, 0, -1):
		x[i-1] = dp[i-1] - cp[i-1] * x[i]
	
	return x


if rank == size - 1:
	propMatOld = np.zeros((int((nX - 1) // size + ((nX - 1) % size)) + 1, nY))
	propMat = np.zeros((int((nX - 1) // size + ((nX - 1) % size)) + 1, nY))
# the last process will account for the residual pieces once they are evenly distributed among all process
else:
	propMatOld = np.zeros((int((nX - 1) // size) + 2, nY))
	propMat = np.zeros((int((nX - 1) // size) + 2, nY))

max_iter = 500 # iteration count
out_iter = 50

count = 0 # counter

# solving the 2D heat equation
while count < max_iter:
	if rank == 0:
		if count % out_iter == 0: print('Iteration ', count, ' DONE, ...\n', flush = True)

		# conditions and parameters for the neumann conditions
		a = - (delX / delY)**2 * np.ones(nY - 1)
		c = a.copy()
		b = delX**2 * gamma * np.ones(nY)
		d = np.zeros(nY)
		for j in range(nY):
			d[j] = 2*propMatOld[1, j]
		propMatOld[0] = triMatAlgo(a, b, c, d) # implementing the neumann conditions, with tridiagonal matrix algo
		propMat[0] = np.copy(propMatOld[0])
	
		comm.Recv(propMatOld[int((nX - 1) // size + 1)], source = rank + 1, tag = 10)
		propMat = np.copy(propMatOld)
		
		# dirichlet BCs
		for i in range(int((nX - 1) // size + 2)):
			propMatOld[i, 0] = 0
			propMat[i, 0] = 0
			propMatOld[i, nY - 1] = 100
			propMat[i, nY - 1] = 100
	
		# with central differences, solving PDE for propMat(i,j)
		# reference, https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
		# reference 2, https://skill-lync.com/projects/week-5-mid-term-project-solving-the-steady-and-unsteady-2d-heat-conduction-problem-35
		for i in range(1,int((nX - 1) // size) + 1):
			for j in range(1, nY - 1):
				propMat[i, j] = ((propMatOld[i+1, j] + propMat[i-1, j]) / delX**2 + (propMatOld[i, j+1] + propMat[i, j-1]) / delY**2) / gamma
		
		comm.Send(propMatOld[int((nX - 1) // size)], dest = rank + 1, tag = 20)
			
		propMatOld = propMat.copy()
		count += 1
		
	elif rank == size - 1:
		comm.Send(propMatOld[1], dest = rank - 1, tag = 10)
		comm.Recv(propMatOld[0], source = rank - 1, tag = 20)
		propMat = np.copy(propMatOld)
		
		# dirichlet BCs
		for i in range(int((nX - 1) // size + ((nX - 1) % size)) + 1):
			propMatOld[i, 0] = 0
			propMat[i, 0] = 0
			propMatOld[i, nY - 1] = 100
			propMat[i, nY - 1] = 100
		
		# with central differences, solving PDE for propMat(i,j)
		for i in range(1,int((nX - 1) // size + ((nX - 1) % size)) - 1):
			for j in range(1, nY - 1):
				propMat[i, j] = ((propMatOld[i+1, j] + propMat[i-1, j]) / delX**2 + (propMatOld[i, j+1] + propMat[i, j-1]) / delY**2) / gamma
		
	
		propMatOld = propMat.copy()
		count += 1
	else:
		comm.Send(propMatOld[1], dest = rank - 1, tag = 10)
		comm.Recv(propMatOld[0], source = rank - 1, tag = 20)
		comm.Recv(propMatOld[int((nX - 1) // size) + 1], source = rank + 1, tag = 10)
		propMat = np.copy(propMatOld)
		
		# dirichlet BCs
		for i in range(int((nX - 1) // size) + 1):
			propMatOld[i, 0] = 0
			propMat[i, 0] = 0
			propMatOld[i, nY - 1] = 100
			propMat[i, nY - 1] = 100
		
		# with central differences, solving PDE for propMat(i,j)
		for i in range(1,int((nX - 1) // size) + 1):
			for j in range(1,nY - 1):
				propMat[i, j] = ((propMatOld[i+1, j] + propMat[i-1, j]) / delX**2 + (propMatOld[i, j+1] + propMat[i, j-1]) / delY**2) / gamma
		
		comm.Send(propMatOld[int((nX - 1) // size)], dest = rank + 1, tag = 20)
		
		propMatOld = propMat.copy()
		count += 1

# removing the edges which would repeat in the compiled matrix is left as it is,
if rank == 0:
	propMat = propMat[:-1, :].copy()
elif rank == size - 1:
	propMat = propMat[1:, :].copy()
else:
	propMat = propMat[:-1, :].copy()
	propMat = propMat[1:, :].copy()

# gathering all the data from the process to compile the PDE
if rank == 0:
	for i in range(1, size):
		if i == size - 1:
			rcv = np.zeros((int((nX - 1) // size + ((nX - 1) % size)), nY))
		else:
			rcv = np.zeros((int((nX - 1) // size), nY))
		comm.Recv(rcv, source = i)
		propMat = np.concatenate((propMat, rcv), axis = 0)
else:
	comm.Send(propMat, dest = 0)

stop_parallel = MPI.Wtime() # stop solving of PDE

time_parallel[0] = stop_parallel - start_parallel
comm.Reduce(time_parallel, total_time, op = MPI.SUM, root = 0)

if rank == 0: # get output for total time elapsed during solving of PDE
	print('Total time for parallel PDE solving: ', total_time[0])

# plotting code
if rank == 0:
	meshX, meshT = np.meshgrid(np.linspace(0, lenX, nX), np.linspace(0, lenY, nY))
	plt.figure()
	plt.contourf(meshX, meshT, np.rot90(propMat, k = 3), 50, cmap='jet')
	plt.colorbar()
	plt.title('Solving 2D PDE Heat Equation', fontsize=13)
	plt.xlabel('x, length', fontsize=12)
	plt.ylabel('y, length', fontsize=12)
	plt.savefig('mm17b024_heat2d.png')
	# plt.show()