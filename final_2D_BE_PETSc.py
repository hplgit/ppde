"""

2-D diffusion equation using Backward Euler (implicit). The diffusion
equation is
	u_t = a*(u_xx + u_yy)
This program is based upon the 1-D solver. The A*un=b system is explained here:
http://pauli.uni-muenster.de/tp/fileadmin/lehre/NumMethoden/WS0910/ScriptPDE/Heat.pdf

Some of the code is inspired by:
https://fs.hlrs.de/projects/par/par_prog_ws/pdf/petsc_exa_heat.pdf

For simplicity the boundaries is zero everywhere. In that case the
right-hand-side b is simply unm1 (the solution at the last time step).

To run this in parallel, use:
mpirun -np <number of processes> python backward_diffusion_2d.py
    (with optional arguments given below: <nx> <ny> <Nt>
This can be run in serial with -np 1 or just run with python, ignoring
mpirun. Note that running this in parallel requires additional
packages in PETSc.
 
This scheme is unconditionally stable.

"""

# Import and initialize
import petsc4py, sys, time
import numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc


def solver_2D_BE_PETSc(I, a, Lx, Ly, Nx, Ny, Cx, Cy, T):
	
	x = np.linspace(0, Lx, Nx+1); y = np.linspace(0, Ly, Ny+1);
	dx = x[1] - x[0]; dy = y[1] - y[0];
	dt = Cx*dx**2/a;   # Based on Cx only
	Nt = int(round(T/float(dt)))
	
	t = np.linspace(0, T, Nt+1)
	t0 = time.clock()
	
	# We need to set up our big matrix, a five-banded sparse MPI mat.
	A = PETSc.Mat().createAIJ([Nx*Ny, Nx*Ny],nnz=5)
	[Istart, Iend] = A.getOwnershipRange()
	for II in range(Istart, Iend):
		i=II/Nx; j=II-i*Nx;
		
		if (i>0):
			J = II-Nx
			A.setValue(II,J,-Cx)
		if (i<Ny-1):
			J = II+Nx
			A.setValue(II,J,-Cx)
		if (j>0):
			J = II-1
			A.setValue(II,J,-Cy)
		if (j<Nx-1):
			J = II+1
			A.setValue(II,J,-Cy)
		A.setValue(II,II,1+2*Cx+2*Cy)
	
	# And then we get our left-handed and right-handed arrays
	u, u_1 = A.getVecs()
	
	"""
	Important that I(x,y) follows the rule of the matrix system. One example is:
	
	for i in range(Nx):
		for j in range(Ny):
			I[i*Ny+j] = math.exp(-((i*dx-0.5)**2 + (j*dy-0.5)**2)*100)
	"""
	u_1.setValues(range(Istart,Iend),I[Istart:Iend])
	#u_1.setArray(I)   # Initialize RHS
	
	# Assemble the arrays and matrix. This distribute all the data.
	A.assemble(); u.assemble(); u_1.assemble()
	
	# If not in parallel, one can write to file for plotting in another
	# program for validation. This is started from the command line using
	# -toFile
	if PETSc.Options().getBool('toFile',default=False):
		if PETSc.COMM_WORLD.getSize() > 1:
			if PETSc.COMM_WORLD.getRank() == 0:
				print 'Warning: not writing to file (using parallel)'
				PETSc.Options().setValue('toFile',False)
		else:
			print 'Writing to file'
			W = PETSc.Viewer().createASCII('test2d_7.txt',format=0)
			u_1.view(W)
	
	# Set up the solver. See the 1D-implementation for a description of this.
	ksp = PETSc.KSP()
	ksp.create()
	ksp.setType('cg')
	ksp.getPC().setType('none')
	ksp.setFromOptions()  # We use command-line, but set cg (no precon) as default
	ksp.setOperators(A)
	
	for t in range(Nt):
		# Solve for next step using the solver set up above using
		# command-line options.
		ksp.solve(u_1,u)
		
		# Could we prevent hard-copying here?
		u.copy(u_1)
		
		# Update if writing to file.
		if PETSc.Options().getBool('toFile',default=False):
			u_1.view(W)
	
	
	return u_1, time.clock()-t0















