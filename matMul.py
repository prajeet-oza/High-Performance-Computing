# ABOUT THE CODE:
# 1) The algorithm is the same the one stated in the notes shared moodle.

# importing relevant libraries
from mpi4py import MPI
import numpy as np
import pprint

comm = MPI.COMM_WORLD

# getting input for rank and size
rank = comm.Get_rank()
size = comm.Get_size()

# defining the matrix A and matrix B dimensions
iA = 150; jA = 50
iB = jA; jB = 200

iC = iA; jC = jB # hence, the dimension of matrix C

start_parallel = MPI.Wtime() # start time for parallel part of the code

# useful constant and variables
aspectRatio = iC / jC
min_diff = np.inf
iProcs = 1; jProcs = 1
time_parallel = np.zeros(1)
total_time_parallel = np.zeros(1)

maxFactor = int(np.sqrt(size)) + 1
procs = np.zeros(2, dtype = int)

if rank == 0: # factorizing the number of processors to get iProcs and jProcs
	# based on the aspect ratio, the factorizing can be split based on i or j
	# where i is in numerator, and j is in denominator
	if aspectRatio > 1:
		for j in range(1, maxFactor):
			ratio = (size / j) / j
			if abs(ratio - aspectRatio) < min_diff:
				min_diff = abs(ratio - aspectRatio)
				jProcs = j
				iProcs = size // j
	else:
		for i in range(1, maxFactor):
			ratio = i / (size / i)
			if abs(ratio - aspectRatio) < min_diff:
				min_diff = abs(ratio - aspectRatio)
				iProcs = i
				jProcs = size // i
	procs = np.array([iProcs, jProcs])
# broadcasting the factors to other processes
comm.Bcast(procs, root = 0)

if rank == 0:
	# defining the A and B matrices
	matA = np.random.rand(iA, jA)
	matB = np.random.rand(iB, jB)
	
	# displaying the matrices
	print('\nMatrix A:')
	print(matA)
	print('\nShape of matrix A: ', matA.shape)

	print('\nMatrix B:')
	print(matB)
	print('\nShape of matrix B: ', matB.shape)

	# splitting the matrices A and B into strips of length iProcs and jProcs
	# unless the dimensions are not completely divisible by iProcs and jProcs
	# and in that not divisible case, the last splits will also contain the leftover rows/columns
	for i in range(iProcs):
		for j in range(jProcs):
			stripA = matA[i * iA // iProcs : (i+1) * iA // iProcs]
			stripB = matB[:, j * jB // jProcs : (j+1) * jB // jProcs]
			if i * jProcs + j == 0:
				localA = stripA.copy()
				localB = stripB.copy()
			else: # sending the strips to other processors, first the size and then the data
				comm.Send(np.array(stripA.shape), dest = i * jProcs + j, tag = 0)
				comm.Send(np.array(stripB.shape), dest = i * jProcs + j, tag = 1)
				comm.Send(np.array(stripA), dest = i * jProcs + j, tag = 65)
				comm.Send(np.array(stripB), dest = i * jProcs + j, tag = 66)
	localC = np.dot(localA, localB) # calculating local C matrix from the strips, for process 0

	matC = np.zeros((iC, jC)) # initialize the matrix C
	for i in range(iProcs):
		for j in range(jProcs):
			if i * jProcs + j == 0: # collecting the data from process 0 from local C matrix
				# matC[i * iC // iProcs : (i+1) * iC // iProcs, j * jC // jProcs : (j+1) * jC // jProcs] = localC.copy()
				rowC = localC.copy()
				# print(rowC.shape)
			else: # collecting the data for C matrix from other processors, first the size and then the data
				shapeC = np.zeros(2, dtype = int)
				comm.Recv(shapeC, source = i * jProcs + j, tag = 2)
				blockC = np.zeros(shapeC)
				comm.Recv(blockC, source = i * jProcs + j, tag = 67)
				# storing the local C matrix into correct locations in matrix C
				# matC[i * iC // iProcs : (i+1) * iC // iProcs, j * jC // jProcs : (j+1) * jC // jProcs] = blockC.copy()
				if j == 0:
					rowC = blockC.copy()
				else:
					rowC = np.hstack((rowC, blockC))
					# print(rowC.shape)

		if i == 0:
			tempC = rowC.copy()
			# print(tempC.shape)
		else:
			tempC = np.vstack((tempC, rowC))
	matC = tempC.copy()
	# printing the matrix C obtained from parallel algorithm
	print('\nMatrix C, from parallel matrix multiplication:')
	print(matC)
	print('\nShape of matrix C, parallel: ', matC.shape)

elif rank < np.prod(procs): # other processors only until the product of iProcs and jProcs are considered
	matA = None
	matB = None

	# receiving the shape of splits of matrix A and B
	shapeA = np.zeros(2, dtype = int)
	shapeB = np.zeros(2, dtype = int)
	comm.Recv(shapeA, source = 0, tag = 0)
	comm.Recv(shapeB, source = 0, tag = 1)
	# receiving the data of splits of matrix A and B
	localA = np.zeros(shapeA)
	localB = np.zeros(shapeB)
	comm.Recv(localA, source = 0, tag = 65)
	comm.Recv(localB, source = 0, tag = 66)

	localC = np.dot(localA, localB) # calculate the local C matrix
	# sending the data of the local C matrix to process 0
	comm.Send(np.array(localC.shape), dest = 0, tag = 2)
	comm.Send(localC, dest = 0, tag = 67)

stop_parallel = MPI.Wtime() # stopping the time for parallel algorithm

# calculating the total time for the algorithm by summing the time for all processors
time_parallel[0] = stop_parallel - start_parallel
comm.Reduce(time_parallel, total_time_parallel, op = MPI.SUM, root = 0)
if rank == 0:
	print('\nTotal time for parallel matrix multiplication: ', total_time_parallel[0])

# calculating the serial matrix multiplication with np.dot()
if rank == 0:
	start_serial = MPI.Wtime()
	C = np.dot(matA, matB)
	print('\nMatrix C, from serial matrix multiplication:')
	print(C)
	print('\nShape of matrix C, serial: ', C.shape)
	stop_serial = MPI.Wtime()
	print('\nTotal time for serial matrix multiplication: ', stop_serial - start_serial)

# showing the difference between the output from both the algorithm
if rank == 0:
	diff_mat = C - matC
	print('\nDifference between the two output C matrix: ')
	print(diff_mat)