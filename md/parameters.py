N_atoms = int(insert) # number of atoms in the system, set to 'insert' for the bash script
start = 1 # defines the system type, 0: simple cubic, 1: fcc, 2: bcc
T_sim = 1.4 # starting reduced temperature of the simulation
rho = 0.5 # reduced density
r_cut = 2.5 # reduced cutoff radius
dT = 0.005 # time step
box_len = (N_atoms / rho)**(1/3) # length of one unit cell box

N_therm = 1 # number of thermalisation steps
N_twrite = 10 # store output at every N_twrite'th step

N_simul = 500 # number of simulation steps
N_swrite = 100 # store output at every N_swrite'th step

# output file name, this will be followed by '_therm' and '_md' based on the sections, effectively generating two output files
outname = 'output'