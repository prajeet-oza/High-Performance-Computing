from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank != 0:
	data = 'DATA PACKET FROM RANK ' + str(rank) + ' TO PROCESS 0'
	comm.send(data, dest = 0, tag = 11)
else:
	for i in range(1, size):
		recv_data = comm.recv(source = i, tag = 11)
		print(recv_data)