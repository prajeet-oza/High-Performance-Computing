# ABOUT THE CODE:
# this is an execution of ring program with mpi4py
# with a trivial attempt to calculate the value of Pi
# calculation of Pi is based on the (area of circle / area of square) approach

# importing relevant libraries
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

# taking input for rank and size
N = comm.Get_size()
rank = comm.Get_rank()

# initialise a counter
value = 0
total = 0
counters = np.array([value, total])

# number of points to consider for each rank
n = 50

# for ranks other than zero, receive value before sending
# or else the counter will be stuck at the initial value
# also, the communication is set to start at 0, and then go clockwise to 1, 2, ... N-1 and back to 0
if rank != 0:
	comm.Recv(counters, source = (rank-1)%N, tag = 13 + (rank-1)%N) # receiving counters from previous rank
	value = counters[0]
	total = counters[1]
	for i in range(n): # considering multiple points at each rank
		x = np.random.uniform(0, 1) # generating random variables x and y from a uniform distribution between 0 and 1
		y = np.random.uniform(0, 1)
		if (x**2 + y**2) <= 1: # checking if the point lies inside the circle or on the circle
			value += 1 # updating if the condition is satisfied
		total += 1 # updating the total number of points
	# printing the value of Pi approximation at each step
	print(' Pi approximation by rank', rank, 'AFTER RECEIVING counters from rank', (rank-1)%N, "with 'value'", value, "and 'total'", total, '= ', 4 * value / total)
	counters = np.array([value, total]) # re-combining the counters and then they can be sent

comm.Send(counters, dest = (rank+1)%N, tag = 13 + rank%N) # sending the counter to the adjacent rank

if rank == 0: # in the end, receiving the counters at rank 0 after the ring is done
	comm.Recv(counters, source = (rank-1)%N, tag = 13 + (rank-1)%N)
	value = counters[0]
	total = counters[1]
	for i in range(n):
		x = np.random.uniform(0, 1)
		y = np.random.uniform(0, 1)
		if (x**2 + y**2) <= 1:
			value += 1
		total += 1
	# final approximation after the ring is over
	print(' Pi approximation by rank', rank, 'AFTER RECEIVING counters from rank', (rank-1)%N, "with 'value'", value, "and 'total'", total, '= ', 4 * value / total)
