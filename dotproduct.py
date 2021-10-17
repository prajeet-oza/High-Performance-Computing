# ABOUT THE CODE:
# The code generates random numbers through gaussian distribution to form the vectors A and B
# and just for checking the correctness of the dot product, a target product is printed using np.dot()
# Now, the numpy array is scattered to the ranks and this is followed by serial dot product.
# This dot product is collected and summed back to rank 0, and the output is printed.
# The dot product is exactly the same as from the np.dot() method.

from mpi4py import MPI # importing relevant libraries
import numpy as np

comm = MPI.COMM_WORLD

# taking input for rank and size
rank = comm.Get_rank()
size = comm.Get_size()

length = 6 # defining the length of each scatter

# setting up the vectors A and B, using rng for gaussian distribution
if rank == 0:
	N = size * length # (size * length) is length of each vector,
	vec_a = np.random.randn(N)
	vec_b = np.random.randn(N)
	print('Vector A: \n', vec_a, '\n') # printing the vectors generated, mostly to cross check the scatter
	print('Vector B: \n', vec_b, '\n')
	# printing the dot product, to cross check the dot product obtained from scatter and reduce
	print('TARGET DOT PRODUCT: ', np.dot(vec_a, vec_b), '# using np.dot() directly on vectors A and B')
else:
	vec_a = None
	vec_b = None

# empty / zeros vectors to store the scatter for each rank
vec_ai = np.zeros(length)
vec_bi = np.zeros(length)

# scattering the two vectors A and B
comm.Scatter(vec_a, vec_ai, root = 0)
comm.Scatter(vec_b, vec_bi, root = 0)

# printing the scatter for each rank, and this scatter can be checked against the original vectors
print('\nRank', rank, 'has section', vec_ai, 'of vector A')
print('Rank', rank, 'has section', vec_bi, 'of vector B \n')

# serial dot product for the scattered vectors, for each rank
doti = np.zeros(1)
for i in range(length):
	doti[0] += vec_ai[i] * vec_bi[i]

# reducing the dot products with sum operation to rank 0
dot_prod = np.zeros(1)
comm.Reduce(doti, dot_prod, op = MPI.SUM, root = 0)

# printing the dot product obtained from scatter and reduce methods, and this can be checked against the previously mentioned target dot product
if rank == 0:
	print('DOT PRODUCT: ', dot_prod[0], '# from Scatter and Reduce methods')