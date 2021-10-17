from mpi4py import MPI
import numpy as np

# function to merge arrays, used with parallel merge sort
def mergeArrays(array1, array2):
	m = array1.shape[0]
	n = array2.shape[0]
	mergedArray = np.zeros(m + n, dtype = int)

	i, j, k = 0, 0, 0

	while i < m and j < n: # lesser value, enter first in the parent array
		if array1[i] <= array2[j]:
			mergedArray[k] = array1[i]
			i += 1
		else:
			mergedArray[k] = array2[j]
			j += 1
		k += 1

	while i < m: # checking if any numbers are left in the child arrays
		mergedArray[k] = array1[i]
		i += 1
		k += 1

	while j < n: # checking if any numbers are left in the child arrays
		mergedArray[k] = array2[j]
		j += 1
		k += 1

	return mergedArray

# function for serial merge sort
def mergeSort(array1):
	if array1.shape[0] > 1:
		mid = array1.shape[0] // 2

		left = array1[:mid].copy() # dividing the array into two parts
		right = array1[mid:].copy()

		left = mergeSort(left) # sorting the two parts
		right = mergeSort(right)

		L = left.shape[0]
		R = right.shape[0]

		i, j, k = 0, 0, 0

		while i < L and j < R: # adding the numbers from the two parts, lesser first into the parent array
			if left[i] <= right[j]:
				array1[k] = left[i]
				i += 1
			else:
				array1[k] = right[j]
				j += 1
			k += 1

		while i < L: # checking if anything is remaining the in the array to be added to the parent array
			array1[k] = left[i]
			i += 1
			k += 1

		while j < R:
			array1[k] = right[j]
			j += 1
			k += 1	

	return array1


comm = MPI.COMM_WORLD

# getting rank and size inputs
rank = comm.Get_rank()
size = comm.Get_size()

length = 2 # length of array in a process, hence the total length is size * length
max_level = np.log2(size) # max level for the tree of processes, with the root as process 0
level = 0 # starting level
time_parallel = np.zeros(1) # variable storing time for each process
total_time = np.zeros(1) # total time after adding time from each process

if rank == 0: # generating random integer array containing size * length numbers
	series = np.random.randint(1, 1000, length * size)
else:
	series = None

# PARALLEL MERGE SORT SECTION OF THE CODE
start_parallel = MPI.Wtime() # starting parallel merge sort

series_local = np.zeros(length, dtype = int) # initialise a variable to obtain scattered array from process 0

comm.Scatter(series, series_local, root = 0) # scattering the array to child processes

if rank == 0: # print the original array
	print('\n' + '*'*10 + ' PARALLEL MERGE SORT ' + '*'*10, flush = True)
	print('\nRandom series generated: ', series, 'and scattered to processes.', flush = True)

# print the scatter received by every process	
print('\nScatter received by rank', rank, ': ', series_local, flush = True)

# sorting the scatter for a process
series_local = np.sort(series_local)

while level < max_level: # collecting/merging the sorted scatter arrays
	# say, at level 0, all the odd processes will merge with the even process to its left
	# say, at level 1, all the processes which are not multiple of 4 will merge with processes divisible by 4, to its left
	# and so on, this exact thing is implemented here in this loop, and breaking when the process is no longer in contention of merging
	if rank % 2**(level + 1) != 0:
		comm.Send(series_local, dest = rank - 2**level)
		break
	else:
		merge = np.copy(series_local)
		comm.Recv(series_local, source = rank + 2**level)
		merge = mergeArrays(merge, series_local)
		series_local = np.copy(merge)
		print('\nLocal series after merging ranks', rank, 'and', rank + 2**level, ': ', series_local) # print the merged array
		level += 1 # incrementing the rank

if rank == 0: # final array obtained after parallel merge sort
	print('\nSeries obtained at rank', rank, 'after parallel merge sort: ', series_local)

stop_parallel = MPI.Wtime() # stop parallel merge sort

time_parallel[0] = stop_parallel - start_parallel
comm.Reduce(time_parallel, total_time, op = MPI.SUM, root = 0)

if rank == 0: # get output for total time elapsed during parallel merge sort
	print('\nTotal time for parallel sorting: ', total_time[0])

# SERIAL MERGE SORT SECTION OF THE CODE, for comparison
if rank == 0: # implementing serial merge sort, hence using only one process
	start_serial = MPI.Wtime() # start serial merge sort

	print('\n' + '*'*10 + ' SERIAL MERGE SORT ' + '*'*10)

	sort_serial = mergeSort(series) # merge sorting the array
	# sort_serial = np.sort(series)

	print('\nSeries obtained after serial merge sort: ', sort_serial) # final array after serial merge sort

	stop_serial = MPI.Wtime() # stop serial merge sort

	time_serial = stop_serial - start_serial # getting output for total time elapsed during serial merge sort
	print('\nTotal time for serial sorting: ', time_serial)
