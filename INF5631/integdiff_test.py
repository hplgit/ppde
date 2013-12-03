# Import and initialize
import petsc4py, sys, time
import numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc

def integdiff_test(a=0, b=1, n=10000000):
	from integdiff import trapezoidal_PETSc, differentiate_PETSc, differentiate_vec
	
	x = np.linspace(a, b, n+1)
	f = 3*x**2   # Function to integrate and differentiate
	
	# Let each rank run trapezoidal function.
	I_PETSc, trap_t0 = trapezoidal_PETSc(f, a, b, n)
	
	# Each rank prints its result (should be the same),
	# and the time it used. The time should be approx. be the same
	if PETSc.COMM_WORLD.getRank() == 0:
		print 'Integrals:\n--------------'
		print 'Rank ('+str(PETSc.COMM_WORLD.getSize())+'):\tTime used:\tIntegral:'
	PETSc.COMM_WORLD.Barrier()
	print PETSc.COMM_WORLD.getRank(), '\t\t', trap_t0, '\t\t', I_PETSc
	
	# Run the PETSc and numpy-vector version. We run a sequential
	# function to check that the solutions is the same.
	diff_PETSc, diff_time = differentiate_PETSc(f, a, b, n)
	diff_vec, diff_time_vec = differentiate_vec(f, a, b, n)
	
	if PETSc.COMM_WORLD.getRank() == 0:
		# We now print the norm (Euclidean) of the difference between the
		# two. If the two functions return the same vectors, the norm should
		# be close to machine precision (~1E-16).
		print '\n\nDiffs:\t\tNorm: ', np.linalg.norm(diff_PETSc-diff_vec)
		print '\nRank:\t\tTime used:\n--------------'
	PETSc.COMM_WORLD.Barrier()
	
	# Print how much time each rank used to run the PETSc function.
	# These should be about the same, but root might use slightly
	# longer time.
	print PETSc.COMM_WORLD.getRank(), '\t\t', diff_time

if __name__ == '__main__':
    integdiff_test()