from final_2D_BE_PETSc import solver_2D_BE_PETSc
# Import and initialize
import petsc4py, sys, time, math
import numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc


def test_2D_BE_PETSc():
	# Get command line options
	a = PETSc.Options().getReal('a',default=1.0)
	Lx = PETSc.Options().getReal('Lx',default=1.0)
	Ly = PETSc.Options().getReal('Ly',default=1.0)
	Nx = PETSc.Options().getInt('Nx',default=100)
	Ny = PETSc.Options().getInt('Ny',default=100)
	Cx = PETSc.Options().getReal('Cx',default=1.0)
	Cy = PETSc.Options().getReal('Cy',default=1.0)
	T = PETSc.Options().getReal('T',default=0.25)
	
	x = np.linspace(0, Lx, Nx); y = np.linspace(0, Ly, Ny);
	dx = x[1] - x[0]; dy = y[1] - y[0];
	I = np.zeros(Nx*Ny)
	for i in range(Nx):
			for j in range(Ny):
				I[i*Ny+j] = math.exp(-((i*dx-0.5)**2 + (j*dy-0.5)**2)*100)

	solution_BE, t_solution_BE = solver_2D_BE_PETSc(I, a, Lx, Ly, Nx, Ny, Cx, Cy, T)

	if PETSc.COMM_WORLD.getRank() == 0:
		print 'Time: ', t_solution_BE

if __name__ == '__main__':
    test_2D_BE_PETSc()