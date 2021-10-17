# importing relevant libraries
import numpy as np

# calling relevant files
from parameters import *
from classes import *

# NOTE 1:
# the supercell size is dependent the number of atoms, and supercell is assumed to be a cube
# i.e., the unit cell is repeated equally in x, y, z directions

# NOTE 2:
# also, the code tries to accomodates for any number of atoms specified in parameters.py,
# which leads to incomplete super cell -- so, it is not practically advised unless some specific cases
# even though i have used incomplete super cells for testing, just to compare the efficiency of the code

# function to initialise position, available options are simple cubic, fcc, bcc
def initPosition(pos):
	i_atom = 0

	if start == 0:
		system = 'simple cubic'
		N_units = int(np.ceil(N_atoms**(1/3)))
		cell_len = box_len / N_units
		# basis for simple cubic is (0, 0, 0)
		for i in range(N_units):
			for j in range(N_units):
				for k in range(N_units):
					if i_atom < N_atoms:
						pos[i_atom] = position()
						coordx = i * cell_len
						coordy = j * cell_len
						coordz = k * cell_len
						pos[i_atom].pos_init(coordx, coordy, coordz)
						i_atom += 1
	elif start == 1:
		system = 'fcc'
		N_units = int(np.ceil((N_atoms/4)**(1/3)))
		cell_len = box_len / N_units
		# basis for fcc is (0, 0, 0), (0.5, 0, 0.5), (0.5, 0.5, 0), (0, 0.5, 0.5)
		for i in range(N_units):
			for j in range(N_units):
				for k in range(N_units):
					if i_atom < N_atoms:
						pos[i_atom] = position()
						coordx = i * cell_len
						coordy = j * cell_len
						coordz = k * cell_len
						pos[i_atom].pos_init(coordx, coordy, coordz)
						i_atom += 1

					if i_atom < N_atoms:
						pos[i_atom] = position()
						coordx = i * cell_len + cell_len / 2
						coordy = j * cell_len + cell_len / 2
						coordz = k * cell_len
						pos[i_atom].pos_init(coordx, coordy, coordz)
						i_atom += 1

					if i_atom < N_atoms:
						pos[i_atom] = position()
						coordx = i * cell_len + cell_len / 2
						coordy = j * cell_len
						coordz = k * cell_len + cell_len / 2
						pos[i_atom].pos_init(coordx, coordy, coordz)
						i_atom += 1

					if i_atom < N_atoms:
						pos[i_atom] = position()
						coordx = i * cell_len
						coordy = j * cell_len + cell_len / 2
						coordz = k * cell_len + cell_len / 2
						pos[i_atom].pos_init(coordx, coordy, coordz)
						i_atom += 1
	elif start == 2:
		system = 'bcc'
		N_units = int(np.ceil((N_atoms/2)**(1/3)))
		cell_len = box_len / N_units
		# basis for bcc is (0, 0, 0) and (0.5, 0.5, 0.5)
		for i in range(N_units):
			for j in range(N_units):
				for k in range(N_units):
					if i_atom < N_atoms:
						pos[i_atom] = position()
						coordx = i * cell_len
						coordy = j * cell_len
						coordz = k * cell_len
						pos[i_atom].pos_init(coordx, coordy, coordz)
						i_atom += 1

					if i_atom < N_atoms:
						pos[i_atom] = position()
						coordx = i * cell_len + cell_len / 2
						coordy = j * cell_len + cell_len / 2
						coordz = k * cell_len + cell_len / 2
						pos[i_atom].pos_init(coordx, coordy, coordz)
						i_atom += 1
	return pos, system

# initialising the velocity, and scaling it to match the specified simulation temperature
def initVelocity(vel):
	vel_sq = 0
	sig_vx, sig_vy, sig_vz = 0, 0, 0

	# initialising the velocity, calculating the mean in each direction, and calculating KE or temperature
	for i in range(N_atoms):
		vel[i] = velocity()
		velx = np.random.random() - 0.5
		vely = np.random.random() - 0.5
		velz = np.random.random() - 0.5
		vel[i].vel_init(velx, vely, velz)
		sig_vx += velx
		sig_vy += vely
		sig_vz += velz
		vel_sq += velx**2 + vely**2 + velz**2

	sig_vx = sig_vx / N_atoms
	sig_vy = sig_vy / N_atoms
	sig_vz = sig_vz / N_atoms

	T_init = vel_sq / (3 * N_atoms)
	# scale to reset the velocity according to the temperature
	scale = np.sqrt(3 * N_atoms * T_sim / vel_sq)

	# scaling the velocities, and calculating the KE/temperature to confirm the scaling outcome
	vel_sq = 0
	for i in range(N_atoms):
		vel[i].vx = (vel[i].vx - sig_vx) * scale
		vel[i].vy = (vel[i].vy - sig_vy) * scale
		vel[i].vz = (vel[i].vz - sig_vz) * scale
		vel_sq += vel[i].vx**2 + vel[i].vy**2 + vel[i].vz**2

	T_reset = vel_sq / (3 * N_atoms)

	return vel, T_init, T_reset

# initializing the force based on the positions initialized earlier in the code
def initForce(pos, ljf):
	# initializing the force variables to zero
	for i in range(N_atoms):
		ljf[i] = lj_force()
		ljf[i].ljf_init(0, 0, 0)

	# defining dummy variables, ri, rij, fi, fij
	ri, rij = common3D(), common3D()
	fi, fij = common3D(), common3D()
	boxBy2 = box_len / 2 # just a useful constant 

	for i in range(N_atoms):
		# storing the current atom in dummy variable
		ri.x, ri.y, ri.z = pos[i].rx, pos[i].ry, pos[i].rz
		fi.x, fi.y, fi.z = ljf[i].fx, ljf[i].fy, ljf[i].fz # basically setting the dummy to zero
		# scaning other atoms in the domain
		for j in range(i+1, N_atoms):
			# applying periodic boundary conditions
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

			# based on cutoff radius, the force calculation will be carried out
			rad_sq = rij.x**2 + rij.y**2 + rij.z**2
			if np.sqrt(rad_sq) < r_cut:
				sigByRad6 = 1 / rad_sq**3
				
				potij = sigByRad6 * (sigByRad6 - 1)
				wrkij = -potij - sigByRad6**2

				f_coeff = -wrkij / rad_sq
				fij.x = f_coeff * rij.x * 24
				fij.y = f_coeff * rij.y * 24
				fij.z = f_coeff * rij.z * 24

				# summing the ij force interaction to fi, i.e. ith atom
				fi.x += fij.x
				fi.y += fij.y
				fi.z += fij.z

				# and subtracting the same force from jth atom, because equal and opposite forces
				ljf[j].fx -= fij.x
				ljf[j].fy -= fij.y
				ljf[j].fz -= fij.z
		# storing the force
		ljf[i].fx = fi.x
		ljf[i].fy = fi.y
		ljf[i].fz = fi.z

	return ljf
