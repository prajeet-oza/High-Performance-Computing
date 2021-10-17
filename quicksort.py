# ALGORITHM:
# 1) Generate the series, scatter it to the processors.
# 2) Now, for the series, pick a pivot and broadcast/send across relevant processors. The pivot is selected at random.
# 3) Based on the pivot, distribute the series into greater, lesser and equal to pivot arrays.
# 4) Send the greater array to higher processors, and lesser arrays to lower processors. The equal arrays are send to the master processor,
# and this acccounts for the multiple occurence of the pivot in the original array.
# 5) Now, repeat steps 2 to 4 recursively on the sub-series that are formed due to splitting based on the pivot.
# 6) Serially sort the sub-series obtained by the processors, and then send this to the master processor.
# 7) Combine the pivots and the gathered sub-series from processors to get the final sorted array.

# importing the relevant libraries
from mpi4py import MPI
import numpy as np

# pivot selecting function
def randomPivot(array1, rank, level):
	temp = 2**(max_level - level)
	pivot = np.zeros(1, dtype = dtype_str)
	pivot[0] = placeholder_pivot
	if rank % temp == 0: # this sets the processors that are allowed to select the pivot based on the level
	# say we have 8 processors to start,
	# first level/split will have process 0 select the pivot and broadcast it
	# second splits will have process 0 and 4 to select and broadcast
	# third splits will have process 0, 2, 4 and 6 to select and broadcast
		if array1.shape[0] != 0:
			pivot = array1[np.random.randint(array1.shape[0])]
			# print(pivot.dtype)
		else: # if the array/series is empty, then the pivot is set as 0 which will be a placeholder
			pivot = pivot[0]
		for i in range(1, int(temp)):
			comm.Send(pivot, dest = rank + i)
	else:
		comm.Recv(pivot, source = rank - rank % temp)
		pivot = pivot[0]

	return pivot

# parallel quick sort function, largely to generate lesser, greater and equal to pivot arrays,
# and then send then up to down to the relevant process
def parallelQS(array1, pivot, rank, level):
	temp = 2**(max_level - level)
	lesserArray = []
	greaterArray = []
	equalArray = []
	delArray = []
	tempArray = []
	# the if-else statement divide the processors based on the previous splitting with pivot
	# lower processors generate greater array,
	# and higher processors generate lesser array, to transmit up and down respectively
	if rank % temp < temp / 2:
		for i in range(array1.shape[0]):
			if array1[i] > pivot: # generate greater array
				greaterArray.append(array1[i])
				delArray.append(i)
			elif array1[i] == pivot: # generate equal array
				equalArray.append(array1[i])
				delArray.append(i)
		for i in range(array1.shape[0]):
			if i not in delArray: # delete the equal and greater terms
				tempArray.append(array1[i])
		array1 = np.copy(np.array(tempArray))

		greaterArray = np.array(greaterArray)

		comm.Send(greaterArray, dest = rank + temp / 2) # sending up to a higher process

		lesserArray = probeRecv(rank + temp / 2, dtype_str) # receiving lower array from a higher process
		array1 = np.append(array1, lesserArray) # append the received array

	else: # similar to the if statement, this is for higher processors to generate lesser array and send down to lower processors
		for i in range(array1.shape[0]):
			if array1[i] < pivot:
				lesserArray.append(array1[i])
				delArray.append(i)
			elif array1[i] == pivot:
				equalArray.append(array1[i])
				delArray.append(i)
		for i in range(array1.shape[0]):
			if i not in delArray:
				tempArray.append(array1[i])
		array1 = np.copy(np.array(tempArray))

		lesserArray = np.array(lesserArray)

		comm.Send(lesserArray, dest = rank - temp / 2)

		greaterArray = probeRecv(rank - temp / 2, dtype_str)
		array1 = np.append(array1, greaterArray)

	array1 = array1.astype(dtype_str)

	return array1, np.array(equalArray) # transmits the equal array, while is then sent to process 0 combined with equal arrays from other processors

# function to execute probe and recv
def probeRecv(src_rank, dtype_str):
	info = MPI.Status()
	comm.Probe(source = src_rank, status = info)
	transmitArray = np.zeros(info.Get_count(MPI.DOUBLE), dtype = dtype_str)
	comm.Recv(transmitArray, source = src_rank)

	return transmitArray # returns the received array


comm = MPI.COMM_WORLD

# getting rank and size input
rank = comm.Get_rank()
size = comm.Get_size()

length = 5 # length of series per processor
max_level = np.log2(size)
level = 0
placeholder_pivot = -5 # ensure that the placeholder is outside the plausible range of random numbers
dtype_str = 'int' # change based on datatype of generated random numbers

total_time = np.zeros(1)
time = np.zeros(1)
start = MPI.Wtime() # starting quick sort time

if rank == 0: # generate series
	series = np.random.randint(1, 1000, length * size)
	init_series = series.copy()
else:
	series = None

series_local = np.zeros(length, dtype = dtype_str)

comm.Scatter(series, series_local, root = 0) # scatter the series

if rank == 0:
	print('\n' + '*'*10 + ' PARALLEL QUICK SORT ' + '*'*10, flush = True)
	print('\nRandom series generated: \n', series, 'and scattered to processes.', flush = True)

print('\nScatter received by rank', rank, ': ', series_local, flush = True)

# defining variable to capture the pivots, and arrange them
pivots = np.zeros(size-1)
equalPivots = np.array([])
tempPivot = np.zeros(1, dtype = dtype_str)

if rank == 0: print('\nNOTE: Pivots denoted as', placeholder_pivot, 'are placeholder pivots, as the sub-series as empty.')
while level < max_level: # recursively, in a loop, select the pivot, and serially sort the arrays for the processors
	pivot = randomPivot(series, rank, level) # generate pivot

	if rank != 0:
		comm.Send(np.array(pivot, dtype = dtype_str), dest = 0) # sending the pivot to master process to document it
	else:
		# at master process, the collected pivots for each splitting are ordered
		# this ordering is useful when combining the pivots and the sorted sub-series obtained from the processors
		levelPivots = np.array(pivot)
		index = [0]
		for i in range(1, size):
			comm.Recv(tempPivot, source = i)
			if tempPivot not in levelPivots:
				index.append(i)
				levelPivots = np.append(levelPivots, tempPivot)
		print('\nPivots at', level+1, 'th spltting of the series: ', levelPivots, flush = True)
		index = np.array(index) + 2**(max_level - level - 1) - 1
		pivots[index.astype(int)] = levelPivots.copy()

	# parallel quick sort, to get the greater and lesser and equal to pivot arrays
	series_local, equalSeries = parallelQS(series_local, pivot, rank, level)

	# combining the series from the sub-series based on the pivot splitting
	# so, after the first split, we get 4 sub-series for a 4 processor system
	# now, this section combines the lower two series and upper two series to give two series
	# which are seperated by the first pivot, and these two combined series get us the second pivots
	if rank % 2**(max_level - level - 1) != 0:
		comm.Send(series_local, dest = rank - rank % 2**(max_level - level - 1))
	else:
		series = np.copy(series_local)
		for i in range(rank + 1, int(rank + 2**(max_level - level - 1))):
			tempArray = probeRecv(i, dtype_str)
			series = np.append(series, tempArray)

	if rank != 0: # sending the equal to pivot array to master process
		comm.Send(equalSeries, dest = 0)
	else: # collecting the equal to pivot arrays and storing them to be dealt with when combining pivots and sub-series
		equalPivots = np.append(equalPivots, equalSeries)
		for i in range(1, size):
			equalTempPivot = probeRecv(i, dtype_str)
			equalPivots = np.append(equalPivots, equalTempPivot)

	level += 1 # increment the level

series = np.sort(series) # serial sorting the series for a process

equalPivots = np.sort(equalPivots) # sorting the equal to pivot arrays at master processor, for easier navigation through the array
# print(equalPivots, pivots)

if rank != 0: # send the serially sorted array to master processor
	comm.Send(series, dest = 0)
else: # at master processor, combine the series and pivots to get the output series
	output_series = np.array(series, dtype = dtype_str)

	for i in range(1, size):
		count = -1 # counter/key to account for the equal to pivot array elements
		tempArray = probeRecv(i, dtype_str) # receive the serially sorted series at master processor
		shape = tempArray.shape[0]

		if pivots[i-1] != placeholder_pivot: # the placeholder is not useful in placing the series and pivots
		# basically, with the placeholder, it says that there is no need to have a pivot and the adjacent arrays are empty
			output_series = np.append(output_series, pivots[i-1]) # add the pivot/partition
			while pivots[i-1] in equalPivots: # now, if there exists an element equal to the just added pivot, check for it's count and add
				if count == -1: # if only one entry in the equal to pivot array, then it is merely the repetition of the pivot entry itself
					equalPivots = np.delete(equalPivots, 0) # hence, delete the entry
					count += 1 # increment the count and check if there are more entries for that pivot
				else: # if more entries are there, then add them to the output series
					output_series = np.append(output_series, pivots[i-1])
					equalPivots = np.delete(equalPivots, 0) # delete those added entries
		output_series = np.append(output_series, tempArray) # append the next serially sorted series

	# show the output and the actual sorted array for comparison		
	print('\nSeries obtained at rank', rank, 'after parallel quick sort: \n', output_series.astype(dtype_str), flush = True)
	print('\nSorted random series for comparison: \n', np.sort(init_series), flush = True)

stop = MPI.Wtime() # stopping the timer for quick sort

time[0] = stop - start
comm.Reduce(time, total_time, op = MPI.SUM, root = 0)
if rank == 0:
	print('\nTotal time for parallel quick sorting: ', total_time[0]) # print the time taken by the sorting