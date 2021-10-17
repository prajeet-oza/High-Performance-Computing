# main.py file (renamed as md_nve_se.py) for SERIAL CODE

# importing relevant libraries
import numpy as np
from mpi4py import MPI # only for time, could have used time but sticking to the mpi4py for homogeneity? Not sure if it that matters. 
# calling relevant files
from parameters import *
from classes import *
from initialise import *
from functions_se import *

start = MPI.Wtime()

# defining the variables to store position, velocity and force data
pos = np.zeros(N_atoms, dtype = position)
vel = np.zeros(N_atoms, dtype = velocity)
ljf = np.zeros(N_atoms, dtype = lj_force)

# initialising the positions, velocity and force
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

# starting the simulation, thermalisation steps if needed or set to 1 in parameters.py
print('\n STARTING THERMALISATION STEPS ...')
msg, pos, vel, ljf = simulation(pos, vel, ljf, N_therm, N_twrite, 0)
print('\n THERMALISATION {}'.format(msg))

# if thermalisation is successful, then the simulation proceeds to the calculations which will be finally considered for results
if msg == 'SIMULATION DONE!!!':
	print('\n STARTING MD STEPS ...')
	msg, pos, vel, ljf = simulation(pos, vel, ljf, N_simul, N_swrite, 1)
	print('\n MD {}\n'.format(msg))
else:
	print('\n MD SIMULATION ABORTED DUE TO ERROR IN THERMALISATION.\n')

stop = MPI.Wtime()

time = stop - start

# printing the time taken by the simulation
print(' TOTAL TIME TAKEN BY THE PROCESS:', time)