# ABOUT THE CODE:
# 1. use periodic boundary condition
# 2. total number of particles should be constant for 'N' iterations
# 3. print the number of particles at the end of the iterations
# 4. track the location of any one particle for the 'N' iterations and show in graph 

# importing relevant libraries
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

# particle object storing the id and location
class particle:
	pId = None
	loc = 0
	def __init__(self, pId, loc):
		self.pId = pId
		self.loc = loc

comm = MPI.COMM_WORLD

# getting input for rank and size
rank = comm.Get_rank()
size = comm.Get_size()

N = 100 # number of particles in the system
n = N // size # number of particles in each process
# integer division to counter the question of indivisibility
# in case of indivisibility, say 100 particles and 6 processes,
# each process will end up with 16 particles, reducing the system count down to 96 from 100

# printing the initial/starting system information
if rank == 0:
	print('Number of particles: ', n * size)
	print('with each process having', n, 'particles.\n', flush = True)
	if N % size != 0:
		print('The total is not', N, 'as it is not divisible by', size)
		print('and hence,', N%size, 'particles are removed.\n', flush = True)

# number of iterations for the random walk
Niters = 50

# defining the domain's start and end boundaries
dom_start = 0
dom_end = 1

# id of particle that is to be tracked for the random walk
trackPId = 5
trackPPath = [] # list to store the path information for the particle

pProcess = [] # list to store the particles in the process
for i in range(n): # initialising the particle data with uniform distribution for the domain
	loci = np.random.random() # np.random.random() is over the half interval [0, 1)
	pProcess.append(particle((n * rank + i), loci)) # the ids range from 0 to (n * size - 1)
	if (n * rank + i) == trackPId:
		trackPPath.append([-1, loci, rank]) # storing the initial position of the particle that is being tracked
# on that note, the data for tracking is stored in the form of: [<ITERATION NUMBER>, <LOCATION OF PARTICLE AT THAT ITERATION>, <PROCESS RANK>]
# and initial iteration number is -1, just to be consistent with the 0-indexing in python

# iterating the particles for each process
for iters in range(Niters):
	delParticle = [] # list to store the indices of particle that leave the process, to go forward or backward
	recvFrwdParticle = [] # particles received by current process that come forward from a back process
	recvBackParticle = [] # particles received by current process that come backward from a front process
	sendFrwdParticle = [] # particles to be sent forward
	sendBackParticle = [] # particles to be sent backward
	totalParticles = np.zeros(1) # total particle counting variables
	for i in range(len(pProcess)): # for each particle in the process, add random increment or decrement between -1 and 1
		pProcess[i].loc += np.random.uniform(-1, 1) # increment or decrement obtained from np.random.uniform() method
		if pProcess[i].loc >= dom_end: # since the initialising RNG is a half interval, the domain end is given a GTE condition
			pProcess[i].loc = pProcess[i].loc - (dom_end - dom_start) # normailising the location to within the domain
			delParticle.append(i) # saving the index if the particle is to sent forward or backward
			sendFrwdParticle.append(pProcess[i]) # saving the particle data appropriately, in forward or backward list
		elif pProcess[i].loc < dom_start:
			pProcess[i].loc = pProcess[i].loc + (dom_end - dom_start)
			delParticle.append(i)
			sendBackParticle.append(pProcess[i])

	tempParticle = [] # temporary list
	for i in range(len(pProcess)): # deleting the particles that leave proces to go forward or backward
		if i not in delParticle:
			tempParticle.append(pProcess[i])
	pProcess = tempParticle.copy()

	comm.send(sendFrwdParticle, dest = (rank + 1)%size) # sending the particles forward or backward
	comm.send(sendBackParticle, dest = (rank - 1)%size)
	recvFrwdParticle = comm.recv(source = (rank - 1)%size) # receiving the particles from forward or backward process
	recvBackParticle = comm.recv(source = (rank + 1)%size)

	for i in range(len(recvFrwdParticle)): # appending the received particles to the particle process list
		pProcess.append(recvFrwdParticle[i])

	for i in range(len(recvBackParticle)):
		pProcess.append(recvBackParticle[i])

	for i in range(len(pProcess)): # tracking the mentioned id particle
		if trackPId == pProcess[i].pId:
			trackPPath.append([iters, pProcess[i].loc, rank]) # storing the data for the particle 

	nParticles = np.zeros(1) # getting the number of particles in the process after the iteration
	nParticles[0] = len(pProcess)
	comm.Reduce(nParticles, totalParticles, op = MPI.SUM, root = 0) # summing the particle count and checking after every iteration

	if rank == 0: # displaying the total particle count
		print('Total particles for iteration', iters, ':', totalParticles[0], flush = True)

print('Process', rank, 'at iteration', iters, 'has', len(pProcess), 'particles') # displaying the particle count for each process at the end of all iterations

# sending the particle path data to process 0, to gather and display
if rank != 0:
	comm.send(trackPPath, dest = 0)
# collecting the particle path data at process 0
if rank == 0:
	trackParticle = [] # list to accumulate the particle path for the mentioned id particle
	for i in range(len(trackPPath)):
		trackParticle.append(trackPPath[i]) # appending the data for process 0
	for i in range(1, size):
		addPPath = comm.recv(source = i) # receiving the appending data from other processes
		for j in range(len(addPPath)):
			trackParticle.append(addPPath[j])
	trackParticle = sorted(trackParticle) # sorting the particle path data based on iteration number
	trackParticle = np.array(trackParticle) # converting to numpy array to ease the splicing?

	y1 = trackParticle[:, 1] # particle path based on location, random walk
	y2 = trackParticle[:, 2] # particle path based on processes, mostly to see the transition of the particle from one process to another
	
	# plotting the random walk, and visualising the process switching for the particle
	img_rw = 'mm17b024_randomWalk_for_particle_' + str(trackPId) + '.png'
	img_process = 'mm17b024_processSwitching_for_particle_' + str(trackPId) + '.png'

	plt.figure()
	plt.plot(y1, label = 'Random Walk')
	plt.title('Random Walk for particle ' + str(trackPId), fontsize=14)
	plt.ylabel('Domain location', fontsize=14)
	plt.xlabel('Iterations', fontsize=14)
	plt.legend()
	plt.savefig(img_rw)

	plt.figure()
	plt.plot(y2, label = 'Process Switching')
	plt.title('Process Switching for particle ' + str(trackPId), fontsize=14)
	plt.ylabel('Processes', fontsize=14)
	plt.xlabel('Iterations', fontsize=14)
	plt.legend()
	plt.savefig(img_process)
