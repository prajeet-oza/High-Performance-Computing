# importing relevant libraries
import numpy as np
from mpi4py import MPI

# calling relevant files
from parameters import *
from classes import *

comm = MPI.COMM_WORLD

# force, potential, work and pressure calculation
def forceCalc(local_pos, local_ljf, pos, rank, size):
	N_local = local_pos.shape[0] # setting up the local particles count
	multCount = N_atoms // size

	for i in range(N_local): # initialise the forces to zero
		local_ljf[i].fx, local_ljf[i].fy, local_ljf[i].fz = 0, 0, 0
	N_cut = 0 # tracking the number of particles screened by cutoff radius
	pot, wrk = 0, 0 # initialise potential and work to zero
	# dummy variables, ri, rij, fi, fij
	ri, rij = common3D(), common3D()
	fi, fij = common3D(), common3D()
	boxBy2 = box_len / 2 # a useful constant

	for i in range(N_local): # starting at ith atom
		# storing ith atom data to dummy variables
		ri.x, ri.y, ri.z = local_pos[i].rx, local_pos[i].ry, local_pos[i].rz
		fi.x, fi.y, fi.z = local_ljf[i].fx, local_ljf[i].fy, local_ljf[i].fz # basically setting the fi dummy to zero
		for j in range(N_atoms): # scanning other atoms in the domain
			if j == ((multCount * rank) + i): # skipping the ith atom in the scan
				continue
			else:
				# periodic boundary condition
				rij.x = ri.x - pos[j].rx
				if rij.x > boxBy2:
					rij.x -= box_len
				elif rij.x < -boxBy2:
					rij.x += box_len

				rij.y = ri.y - pos[j].ry
				if rij.y > boxBy2:
					rij.y -= box_len
				elif rij.y < -boxBy2:
					rij.y += box_len

				rij.z = ri.z - pos[j].rz
				if rij.z > boxBy2:
					rij.z -= box_len
				elif rij.z < -boxBy2:
					rij.z += box_len

				rad_sq = rij.x**2 + rij.y**2 + rij.z**2
				if np.sqrt(rad_sq) < r_cut: # screening the distances based on cutoff radius
					sigByRad6 = 1 / rad_sq**3
					
					potij = sigByRad6 * (sigByRad6 - 1) # ij potential
					wrkij = -potij - sigByRad6**2 # ij work
					# accounting for double sum, and adding the potential and work
					pot += (potij / 2)
					wrk += (wrkij / 2)

					f_coeff = -wrkij / rad_sq
					fij.x = f_coeff * rij.x * 24
					fij.y = f_coeff * rij.y * 24
					fij.z = f_coeff * rij.z * 24
					# summing the forces for ith atom
					fi.x += fij.x
					fi.y += fij.y
					fi.z += fij.z

					N_cut += 1 # counting the cutoff atoms, for truncated potential
		# storing the forces for ith atom
		local_ljf[i].fx = fi.x
		local_ljf[i].fy = fi.y
		local_ljf[i].fz = fi.z

	# accounting for the truncated potential
	if rank == 0:
		N_cut_total = N_cut
		for i in range(1, size):
			temp_N_cut = comm.recv(source = i, tag = 35)
			N_cut_total += temp_N_cut
	else:
		comm.send(N_cut, dest = 0, tag = 35)

	if rank == 0:
		sigByRad3 = 1 / r_cut**3
		sigByRad6 = 1 / r_cut**6
		potij = sigByRad6 * (sigByRad6 - 1)
		pot -= (potij * N_cut_total / 2)

	pot *= 4
	wrk *= 8

	# long range correction terms
	if rank == 0:
		pot_corr = (8 / 9) * np.pi * rho * N_atoms * (sigByRad3**3 - 3 * sigByRad3)
		wrk_corr = (16 / 9) * np.pi * rho**2 * N_atoms * (2 * (sigByRad3**3) - 3 * sigByRad3)	
		pot += pot_corr
		wrk += wrk_corr

	# hence, calculating pressure from work
	if rank == 0:
		prs = (N_atoms * T_sim + wrk) / box_len**3
	else:
		prs = wrk / box_len**3

	return local_ljf, pot, prs

# implementing velocity verlet algorithm
def velVerlet(local_pos, local_vel, local_ljf, rank, size):
	N_local = local_pos.shape[0]

	for i in range(N_local):
		# position update and periodic boundary condition
		local_pos[i].rx += dT * local_vel[i].vx + dT**2 * local_ljf[i].fx / 2
		if local_pos[i].rx >= box_len:
			local_pos[i].rx -= box_len
		elif local_pos[i].rx < 0:
			local_pos[i].rx += box_len

		local_pos[i].ry += dT * local_vel[i].vy + dT**2 * local_ljf[i].fy / 2
		if local_pos[i].ry >= box_len:
			local_pos[i].ry -= box_len
		elif local_pos[i].ry < 0:
			local_pos[i].ry += box_len

		local_pos[i].rz += dT * local_vel[i].vz + dT**2 * local_ljf[i].fz / 2
		if local_pos[i].rz >= box_len:
			local_pos[i].rz -= box_len
		elif local_pos[i].rz < 0:
			local_pos[i].rz += box_len

		# half step velocity update
		local_vel[i].vx += dT * local_ljf[i].fx / 2
		local_vel[i].vy += dT * local_ljf[i].fy / 2
		local_vel[i].vz += dT * local_ljf[i].fz / 2

	# calculating time spent in communication, P1 approach
	startv = MPI.Wtime()
	pos = parallelise(local_pos, rank, size)
	stopv = MPI.Wtime()
	timev = stopv - startv
	# calculating force based on position update
	local_ljf, local_pot, local_prs = forceCalc(local_pos, local_ljf, pos, rank, size)

	for i in range(N_local): # full step velocity update
		local_vel[i].vx += dT * local_ljf[i].fx / 2
		local_vel[i].vy += dT * local_ljf[i].fy / 2
		local_vel[i].vz += dT * local_ljf[i].fz / 2

	return local_pos, local_vel, local_ljf, local_pot, local_prs, timev

# kinetic energy and temperature calculation
def keCalc(local_vel):
	vel_sq = 0
	N_local = local_vel.shape[0]
	for i in range(N_local):
		vel_sq += local_vel[i].vx**2 + local_vel[i].vy**2 + local_vel[i].vz**2
	local_kin = vel_sq / 2 # KE
	local_T_upd =  vel_sq / (3 * N_atoms) # temperature

	return local_kin, local_T_upd

# writing the output file
def outputFile(t, KE, PE, T, Pressure, filename):
	try:
		file1 = open(filename, 'r+')
		read = file1.readlines()
		# storing kinetic energy, potential energy, total energy, temperature, presssure
		if t == 0:
			line = 'Time\tKE\tPE\tTE\tTemp\tPres\n'
			file1.writelines(line)
			line = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(str(t), str(KE), str(PE), str(KE+PE), str(T), str(Pressure))
			file1.writelines(line)
		else:
			line = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(str(t), str(KE), str(PE), str(KE+PE), str(T), str(Pressure))
			file1.writelines(line)

		file1.close()

		return 'added'
	except:
		return 'abort'

# controlling the steps in simulation
def simulation(local_pos, local_vel, local_ljf, N_steps, N_out, index, pdata):
	rank, size = pdata

	# creating the output file
	if rank == 0:
		if index == 0:
			filename = outname + '_therm.dat'
		elif index == 1:
			filename = outname + '_md.dat'

		file1 = open(filename, 'w+')
		file1.close()

	# simulation for N_steps
	time = 0
	total_time = 0
	for i in range(N_steps):
		try:
			# calculating the time taken in communication, P1 approach
			start = MPI.Wtime()
			pos = parallelise(local_pos, rank, size) # getting all the atom data
			stop = MPI.Wtime()
			time = stop - start
			total_time += time
			# velocity verlet implementation
			local_pos, local_vel, local_ljf, local_pot, local_prs, timev = velVerlet(local_pos, local_vel, local_ljf, rank, size)
			# kinetic energy and temperature calculation
			local_kin, local_T_upd = keCalc(local_vel)
			comm.Barrier() # synchronise

			total_time += timev

			# combining all the local property values
			kin = comm.reduce(local_kin, op = MPI.SUM, root = 0)
			pot = comm.reduce(local_pot, op = MPI.SUM, root = 0)
			prs = comm.reduce(local_prs, op = MPI.SUM, root = 0)
			T_upd = comm.reduce(local_T_upd, op = MPI.SUM, root = 0)

			# writing to the output file
			if rank == 0:
				if i % N_out == 0:
					try:
						status = outputFile(i, kin, pot, T_upd, prs, filename)
						if status == 'added':
							print('\t Step {} done'.format(i), flush = True)
						elif status == 'abort':
							return 'SIMULATION ABORTED!!! ERROR IN OUTPUT FILE.', local_pos, local_vel, local_ljf
					except:
						return 'SIMULATION ABORTED!!! ERROR IN OUTPUT FILE.', local_pos, local_vel, local_ljf
		except:
			return 'SIMULATION ABORTED!!! ERROR IN CALCULATION.', local_pos, local_vel, local_ljf
	if rank == 0: print(' COMMUNICATION TIME: ', total_time, flush = True)
	return 'SIMULATION DONE!!!', local_pos, local_vel, local_ljf

# P1 approaching collection of all atom data
def parallelise(local_pos, rank, size):
	pos = np.zeros(N_atoms, dtype = position)

	if rank == 0: # collecting all the data at root node
		N_local = N_atoms // size
		for i in range(size):
			if i == 0:
				pos[i*N_local:(i+1)*N_local] = local_pos.copy()
			elif i == size - 1:
				temp_pos = np.zeros(N_local + (N_atoms % size), dtype = position)
				temp_pos = comm.recv(source = i, tag = 10)
				pos[i*N_local:] = temp_pos.copy()
			else:
				temp_pos = np.zeros(N_local, dtype = position)
				temp_pos = comm.recv(source = i, tag = 10)
				pos[i*N_local:(i+1)*N_local] = temp_pos.copy()
	else: # trasmitting the data to the root node
		comm.send(local_pos, dest = 0, tag = 10)
		
	# broadcasting the all atom data from root
	pos = comm.bcast(pos, root = 0)

	return pos