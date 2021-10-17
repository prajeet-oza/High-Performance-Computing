# main.py file (renamed as md_nve_p1.py) for P1 DECOMPOSITION

# importing relevant libraries
import numpy as np
from mpi4py import MPI
# calling relevant files
from parameters import *
from classes import *
from initialise import *
from functions_p1 import *

comm = MPI.COMM_WORLD
# getting rank and size data
rank = comm.Get_rank()
size = comm.Get_size()
# starting the timer
start = MPI.Wtime()
time = 0

# defining the variables to store position, velocity and force data
pos = np.zeros(N_atoms, dtype = position)
vel = np.zeros(N_atoms, dtype = velocity)
ljf = np.zeros(N_atoms, dtype = lj_force)

# defining the size of particles each processor will handle, and hence, local data variables
if rank == size - 1:
	N_local = (N_atoms // size) + (N_atoms % size)
else:
	N_local = N_atoms // size
local_pos = np.zeros(N_local, dtype = position)
local_vel = np.zeros(N_local, dtype = velocity)
local_ljf = np.zeros(N_local, dtype = lj_force)

# initialising the positions, velocity and force
if rank == 0:
	pos, system = initPosition(pos)
	vel, T_init, T_reset = initVelocity(vel)
	ljf = initForce(pos, ljf)
	# displaying the information related to initialised system
	print('\n SYSTEM INITIALISED!!! (REDUCED LENNARD JONES UNITS.)', flush = True)
	print(' Details of the system ' + 5*'=' + '>', flush = True)
	print('\t Number of atoms: {}, Density: {}'.format(N_atoms, rho), flush = True)
	print('\t Cutoff radius: {}, Time step: {}'.format(r_cut, dT), flush = True)
	print('\t Initialised system configuration: {}'.format(system), flush = True)
	print('\n')
	print('\t Temperature at velocity initialisation: {}'.format(T_init), flush = True)
	print('\t Temperature after resetting and scaling velocity: {}'.format(T_reset), flush = True)
	print('\n')
	print('\t Thermalisation steps to be done: {}'.format(N_therm), flush = True)
	print('\t Simulation steps to done: {}'.format(N_simul), flush = True)

	# splitting the particles data, after initialisation from the root node to other nodes
	for i in range(size):
		if i == 0:
			local_pos = pos[i*N_local:(i+1)*N_local].copy()
			local_vel = vel[i*N_local:(i+1)*N_local].copy()
			local_ljf = ljf[i*N_local:(i+1)*N_local].copy()
		elif i == size - 1:
			temp_pos = pos[i*N_local:].copy()
			temp_vel = vel[i*N_local:].copy()
			temp_ljf = ljf[i*N_local:].copy()

			comm.send(temp_pos, dest = i, tag = 1001)
			comm.send(temp_vel, dest = i, tag = 1002)
			comm.send(temp_ljf, dest = i, tag = 1003)
		else:
			temp_pos = pos[i*N_local:(i+1)*N_local].copy()
			temp_vel = vel[i*N_local:(i+1)*N_local].copy()
			temp_ljf = ljf[i*N_local:(i+1)*N_local].copy()
			
			comm.send(temp_pos, dest = i, tag = 1001)
			comm.send(temp_vel, dest = i, tag = 1002)
			comm.send(temp_ljf, dest = i, tag = 1003)
else:
	local_pos = comm.recv(source = 0, tag = 1001)
	local_vel = comm.recv(source = 0, tag = 1002)
	local_ljf = comm.recv(source = 0, tag = 1003)

# starting the simulation, thermalisation steps if needed or set to 1 in parameters.py
if rank == 0:
	print('\n STARTING THERMALISATION STEPS ...', flush = True)

msg, local_pos, local_vel, local_ljf = simulation(local_pos, local_vel, local_ljf, N_therm, N_twrite, 0, [rank, size])

# displaying any messages related to the simulation, and errors if any
if rank == 0:
	comb_msg = [msg]
	for i in range(1, size):
		temp_msg = comm.recv(source = i, tag = 2)
		comb_msg.append(temp_msg)
	if 'SIMULATION DONE!!!' in np.unique(comb_msg) and np.unique(comb_msg).shape[0] == 1:
		msg = 'SIMULATION DONE!!!'
		print('\n THERMALISATION {}'.format(msg), flush = True)
	else:
		msg = 'SIMULATION ABORTED!!! ERROR IN THERMALISATION.'
		print('\n THERMALISATION {}'.format(msg), flush = True)
else:
	comm.send(msg, dest = 0, tag = 2)

# if thermalisation is successful, then the simulation proceeds to the calculations which will be finally considered for results
if msg == 'SIMULATION DONE!!!':
	if rank == 0: print('\n STARTING MD STEPS ...', flush = True)

	msg, local_pos, local_vel, local_ljf = simulation(local_pos, local_vel, local_ljf, N_simul, N_swrite, 1, [rank, size])

	# displaying any messages related to the simulation, and errors if any
	if rank == 0:
		comb_msg = [msg]
		for i in range(1, size):
			temp_msg = comm.recv(source = i, tag = 2)
			comb_msg.append(temp_msg)
		if 'SIMULATION DONE!!!' in np.unique(comb_msg) and np.unique(comb_msg).shape[0] == 1:
			print('\n MD {}'.format(msg), flush = True)
		else:
			msg = 'SIMULATION ABORTED!!! ERROR IN MD.'
			print('\n MD {}'.format(msg), flush = True)
	else:
		comm.send(msg, dest = 0, tag = 2)
else:
	print('\n MD SIMULATION ABORTED DUE TO ERROR IN THERMALISATION.\n', flush = True)
	print(' ERROR IN RANK {}'.format(rank), flush = True)

stop = MPI.Wtime()

time = stop - start

# printing the time taken by the simulation
if rank == 0:
	print(' TOTAL TIME TAKEN BY THE PROCESS:', time)