import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

delX = 0.1
delT = 50
alpha = 1 / 10**4
gamma = alpha * delT / delX**2

length = 10
Nsteps = 10000
time = Nsteps * delT

propL, propR = 0, 100

J_root = int(length / delX)

J_local = J_root // size
if rank == size - 1:
	J_local = J_root // size + J_root % size + 1

N_root = int(time / delT)

prop_root = np.zeros((N_root + 1, J_root + 1))
prop_local = np.zeros((N_root, J_local))
print(prop_local.shape, rank)
minX_local = rank * J_root // size * delX
maxX_local = (rank + 1) * J_root // size * delX + (J_local - J_root // size) * delX
print(minX_local, maxX_local, rank)

x_local = np.arange(minX_local, maxX_local, delX)
print(x_local.shape)
propInit_local = 0 * x_local
prop_local[0] = propInit_local.copy()

if rank == 0:
	x = np.arange(0, length + delX, delX)
	propInit = 0 * x
	prop_root[0] = propInit.copy()
	prop_root[0, 0] = propL
	prop_root[0, -1] = propR

	prop_local[:, 0] = propL
elif rank == size - 1:
	prop_local[:, -1] = propR


if rank == 0 or rank == size - 1:
	J_temp = J_local - 1
else:
	J_temp = J_local

p = (1 + 2 * gamma) * np.ones(J_temp)
q = -1 * gamma * np.ones(J_temp - 1)
triMat = np.diag(p) + np.diag(q, k = -1) + np.diag(q, k = 1)
invMat = np.linalg.inv(triMat)
# print(triMat.shape, rank)
for t in range(N_root - 1):
	temp = np.zeros(J_temp)
	sendL = np.zeros(1)
	sendR = np.zeros(1)
	recvL = np.zeros(1)
	recvR = np.zeros(1)

	if rank == 0:
		# sendR[0] = prop_local[t+1, -1]
		# comm.Send(sendR, dest = rank + 1, tag = 20)
		# comm.Recv(recvR, source = rank + 1, tag = 10)
		temp[0] = prop_local[t+1, 0]
		# temp[-1] = recvR[0]
	elif rank == size - 1:
		# sendL[0] = prop_local[t+1, 0]
		# comm.Send(sendL, dest = rank - 1, tag = 10)
		# comm.Recv(recvL, source = rank - 1, tag = 20)
		# temp[0] = recvL[0]
		temp[-1] = prop_local[t+1, -1]
	# else:
	# 	sendL[0] = prop_local[t+1, 0]
	# 	sendR[0] = prop_local[t+1, -1]
	# 	comm.Send(sendL, dest = rank - 1, tag = 10)
	# 	comm.Send(sendR, dest = rank + 1, tag = 20)
	# 	comm.Recv(recvR, source = rank + 1, tag = 10)
	# 	comm.Recv(recvL, source = rank - 1, tag = 20)
	# 	temp[0] = recvL[0]
	# 	temp[-1] = recvR[0]
	temp = gamma * temp.copy()

	if rank == 0:
		# mulMat = np.dot(triMat, prop_local[t, 1:].T)
		addMat = prop_local[t, 1:].T + temp.T
		prop_local[t+1, 1:] = np.dot(invMat, addMat)
	elif rank == size - 1:
		# mulMat = np.dot(triMat, prop_local[t, :-1].T)
		addMat = prop_local[t, :-1].T + temp.T
		prop_local[t+1, :-1] = np.dot(invMat, addMat)
	else:
		# mulMat = np.dot(triMat, prop_local[t, :].T)
		addMat = prop_local[t, :].T + temp.T
		prop_local[t+1, :] = np.dot(invMat, addMat)

# print(prop_local.shape, rank)

if rank == 0:
	prop_root[1:, rank * J_local : (rank+1) * J_local] = prop_local.copy()
	for proc in range(1, size):
		if proc == size - 1:
			recvArray = np.zeros((N_root, J_local + J_root % size + 1))
			comm.Recv(recvArray, source = proc)
			# print(recvArray.shape)
			prop_root[1:, proc * J_local:] = recvArray.copy()
		else:
			recvArray = np.zeros((N_root, J_local))
			comm.Recv(recvArray, source = proc)
			prop_root[1:, proc * J_local : (proc+1) * J_local] = recvArray.copy()
else:
	comm.Send(prop_local, dest = 0)
	# print(prop_local.shape, rank)

if rank == 0:
	meshX, meshT = np.meshgrid(np.arange(0, length + delX, delX), np.arange(0, time + delT, delT))
	title = 'Implicit Finite Difference Scheme with alpha = ' + str('%.2f' % gamma)
	plt.figure()
	plt.contourf(meshX, meshT, prop_root, 50, cmap='jet')
	plt.colorbar()
	plt.suptitle(title, fontsize=14, y=0.97)
	plt.title('with Dirichlet Boundary Conditions', fontsize=13)
	plt.xlabel('x; space mesh', fontsize=12)
	plt.ylabel('t; time mesh || delT * numTimeStep', fontsize=12)
	plt.savefig(str(rank+1)+'.png')
	# plt.show()

# delX = 0.01
# delT = 0.1
# numTimeStep = 100000
# minX, maxX = 0, 1
# minT, maxT = 0, delT*numTimeStep
# D = 1/10**4
# alpha = (D * delT) / delX**2

# # J for space, N for time
# J = int((maxX - minX) / delX + 1)
# N = int((maxT - minT) / delT + 1)

# # initialize the NxJ matrix for u
# u = np.zeros((N, J))

# # boundary conditions (dirichlet); CHANGE AS NEEDED
# minBC, maxBC = 0, 100
# u[:, 0] = minBC
# u[:, J-1] = maxBC