# refer functions_p1.py or functions_p2.py for comments

import numpy as np

from parameters import *
from classes import *

def forceCalc(pos, ljf):
	for i in range(N_atoms):
		ljf[i].fx, ljf[i].fy, ljf[i].fz = 0, 0, 0
	N_cut = 0
	pot, wrk = 0, 0
	ri, rij = common3D(), common3D()
	fi, fij = common3D(), common3D()
	boxBy2 = box_len / 2

	for i in range(N_atoms):
		ri.x, ri.y, ri.z = pos[i].rx, pos[i].ry, pos[i].rz
		fi.x, fi.y, fi.z = ljf[i].fx, ljf[i].fy, ljf[i].fz
		for j in range(i+1, N_atoms):
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
			if np.sqrt(rad_sq) < r_cut:
				sigByRad6 = 1 / rad_sq**3
				
				potij = sigByRad6 * (sigByRad6 - 1)
				wrkij = -potij - sigByRad6**2
				pot += potij
				wrk += wrkij

				f_coeff = -wrkij / rad_sq
				fij.x = f_coeff * rij.x * 24
				fij.y = f_coeff * rij.y * 24
				fij.z = f_coeff * rij.z * 24

				fi.x += fij.x
				fi.y += fij.y
				fi.z += fij.z

				ljf[j].fx -= fij.x
				ljf[j].fy -= fij.y
				ljf[j].fz -= fij.z
				N_cut += 1

		ljf[i].fx = fi.x
		ljf[i].fy = fi.y
		ljf[i].fz = fi.z

	sigByRad3 = 1 / r_cut**3
	sigByRad6 = 1 / r_cut**6
	potij = sigByRad6 * (sigByRad6 - 1)
	pot -= potij * N_cut

	pot *= 4
	wrk *= 8
	pot_corr = (8 / 9) * np.pi * rho * N_atoms * (sigByRad3**3 - 3 * sigByRad3)
	wrk_corr = (16 / 9) * np.pi * rho**2 * N_atoms * (2 * (sigByRad3**3) - 3 * sigByRad3)
	
	pot += pot_corr
	wrk += wrk_corr

	prs = (N_atoms * T_sim + wrk) / box_len**3

	return ljf, pot, prs


def velVerlet(pos, vel, ljf):
	for i in range(N_atoms):
		pos[i].rx += dT * vel[i].vx + dT**2 * ljf[i].fx / 2
		if pos[i].rx >= box_len:
			pos[i].rx -= box_len
		elif pos[i].rx < 0:
			pos[i].rx += box_len

		pos[i].ry += dT * vel[i].vy + dT**2 * ljf[i].fy / 2
		if pos[i].ry >= box_len:
			pos[i].ry -= box_len
		elif pos[i].ry < 0:
			pos[i].ry += box_len

		pos[i].rz += dT * vel[i].vz + dT**2 * ljf[i].fz / 2
		if pos[i].rz >= box_len:
			pos[i].rz -= box_len
		elif pos[i].rz < 0:
			pos[i].rz += box_len

		vel[i].vx += dT * ljf[i].fx / 2
		vel[i].vy += dT * ljf[i].fy / 2
		vel[i].vz += dT * ljf[i].fz / 2

	ljf, pot, prs = forceCalc(pos, ljf)

	for i in range(N_atoms):
		vel[i].vx += dT * ljf[i].fx / 2
		vel[i].vy += dT * ljf[i].fy / 2
		vel[i].vz += dT * ljf[i].fz / 2

	return pos, vel, ljf, pot, prs


def keCalc(vel):
	vel_sq = 0
	for i in range(N_atoms):
		vel_sq += vel[i].vx**2 + vel[i].vy**2 + vel[i].vz**2
	kin = vel_sq / 2
	T_upd =  vel_sq / (3 * N_atoms)

	return kin, T_upd


def outputFile(t, KE, PE, T, Pressure, filename):
	try:
		file1 = open(filename, 'r+')
		read = file1.readlines()

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


def simulation(pos, vel, ljf, N_steps, N_out, index):
	if index == 0:
		filename = outname + '_therm.dat'
	elif index == 1:
		filename = outname + '_md.dat'

	file1 = open(filename, 'w+')
	file1.close()

	for i in range(N_steps):
		try:
			pos, vel, ljf, pot, prs = velVerlet(pos, vel, ljf)
			kin, T_upd = keCalc(vel)
			if i % N_out == 0:
				try:
					status = outputFile(i, kin, pot, T_upd, prs, filename)
					if status == 'added':
						print('\t Step {} done'.format(i), flush = True)
					elif status == 'abort':
						return 'SIMULATION ABORTED!!! ERROR IN OUTPUT FILE.', pos, vel, ljf
				except:
					return 'SIMULATION ABORTED!!! ERROR IN OUTPUT FILE.', pos, vel, ljf
		except:
			return 'SIMULATION ABORTED!!! ERROR IN CALCULATION.', pos, vel, ljf
	return 'SIMULATION DONE!!!', pos, vel, ljf