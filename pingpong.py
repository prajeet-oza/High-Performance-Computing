# ABOUT THE CODE:
# Counter initialises to -1, and then the ping pong starts to pass
# from 0 to 1 to back to 0 and then to 1 and continues, and incrementing the counter every time it is received.

from mpi4py import MPI # importing the libraries

comm = MPI.COMM_WORLD

rank = comm.Get_rank() # getting the rank of the process

count = -1 # initialise the count to -1

passes = 6 # define the number of passes

if rank == 0: # for rank 0, start with sending, and then receive the ping pong for the said number of passes
	for i in range(passes // 2): # send / receive for the given passes
		count += 1 # incrementing the count
		comm.isend(count, dest = 1, tag = 1) # sending to rank 1
		req = comm.irecv(source = 1, tag = 2) # receiving from rank 1
		count = req.wait() # waiting to confirm
		print('COUNT = ', count,'~ PING PONG received at rank', rank, 'from rank 1') # printing the message

if rank == 1: # for rank 1, start with receiving the ping pong and continue to send it back to rank 0 for the said number of passes
	for i in range(passes // 2): # receive / send for the given passes
		req = comm.irecv(source = 0, tag = 1) # receiving from rank 0
		count = req.wait() # waiting to confirm
		print('COUNT = ', count,'~ PING PONG received at rank', rank, 'from rank 0') # printing the message
		count += 1 # incrementing the count
		comm.isend(count, dest = 0, tag = 2) # sending to rank 0