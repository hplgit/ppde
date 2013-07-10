# Import and initialize
import petsc4py, sys, time
import numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc

a = 0; b = 1; n = 10000000

from integdiff import trapezoidal_PETSc, differentiate_PETSc, differentiate_vec
t0 = time.clock()

x = np.linspace(a, b, n+1)
f = 3*x**2
I_PETSc = trapezoidal_PETSc(f, a, b, n)
if PETSc.COMM_WORLD.getRank() == 0:
	print I_PETSc


#x = np.linspace(a, b, n+1)
#f = 3*x**2
#diff_PETSc = differentiate_PETSc(f, a, b, n)
#diff_vec = differentiate_vec(f, a, b, n)

'''
if PETSc.COMM_WORLD.getRank() == 0:
	print 'norm=',np.linalg.norm(diff_PETSc-diff_vec)
	if diff_vec.size < 10:
		print diff_PETSc
		print diff_vec
		print diff_PETSc-diff_vec
'''

if PETSc.COMM_WORLD.getRank() == 0:
	print n, time.clock()-t0