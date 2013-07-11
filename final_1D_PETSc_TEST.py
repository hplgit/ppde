# Run using e.g.:   (Stausland)
# /usr/lib64/openmpi/1.4-gcc/bin/mpirun -np 1 python2.6 final_1D_PETSc_TEST.py -ksp_type cg -pc_type  -Nx 100 -T 0.5
# /usr/lib64/openmpi/1.4-gcc/bin/mpirun -np 1 python2.6 final_1D_PETSc_TEST.py -ksp_type cg -pc_type none -Nx 200 -draw

from final_1D_PETSc import solver_FE_MAT_PETSc, solver_FE_MATfree_PETSc, solver_BE_PETSc
# Import and initialize
import petsc4py, sys, time
import numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc


def test_all_1D_PETSc():
	
	# Read the command line options
	a = PETSc.Options().getReal('a',default=1.0)
	L = PETSc.Options().getReal('L',default=1.0)
	Nx = PETSc.Options().getInt('Nx',default=100)
	C = PETSc.Options().getReal('C',default=0.5)
	T = PETSc.Options().getReal('T',default=0.25)

	x = np.linspace(0,L,Nx+1)
	I = np.exp(-np.square(x-0.5*L)*50)
	
	# Run each
	solution_fe_mat, t_time_fe_mat = solver_FE_MAT_PETSc(I, a, L, Nx, C, T)
	solution_fe_matfree, t_time_fe_matfree = solver_FE_MATfree_PETSc(I, a, L, Nx, C, T)
	solution_be, t_time_be = solver_BE_PETSc(I, a, L, Nx, C, T)
	if PETSc.COMM_WORLD.getRank() == 0:
		print 'uFE1=FE mat \t\tuFE2=FE Matfree \t uBE=BE\n------------'
		print 'Norm (uFE1 - uFE2):\t', np.linalg.norm(solution_fe_mat-solution_fe_matfree)
		print 'Norm (uFE1 - uBE):\t', np.linalg.norm(solution_fe_mat-solution_be)
		print 'Norm (uFE2 - uBE):\t', np.linalg.norm(solution_fe_matfree-solution_be)
		print '\n------------'
		print 'Rank ('+str(PETSc.COMM_WORLD.getSize())+'):\tT(uFE1) \tT(uFE2) \tT(uBE)'
	PETSc.COMM_WORLD.Barrier()
	
	print PETSc.COMM_WORLD.getRank(),'\t\t', t_time_fe_mat, '\t\t', t_time_fe_matfree, '\t\t', t_time_be
	
	PETSc.COMM_WORLD.Barrier()
	if PETSc.COMM_WORLD.getRank() == 0:
		print '\n'

if __name__ == '__main__':
	test_all_1D_PETSc()