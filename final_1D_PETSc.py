"""
Functions for solving a 1D diffusion equations of simplest types
(constant coefficient, no source term):

      u_t = a*u_xx on (0,L)

with boundary conditions u=0 on x=0,L, for t in (0,T].
Initial condition: u(x,0)=I(x).

The following naming convention of variables are used.

===== ==========================================================
Name  Description
===== ==========================================================
Nx    The total number of mesh cells; mesh points are numbered
      from 0 to Nx.
C     The dimensionless number a*dt/dx**2, which implicitly
      specifies the time step.
T     The stop time for the simulation.
I     Initial condition (Python function of x).
a     Variable coefficient (constant).
L     Length of the domain ([0,L]).
x     Mesh points in space.
t     Mesh points in time.
n     Index counter in time.
u     Unknown at current/new time level.
u_1   u at the previous time level.
dx    Constant mesh spacing in x.
dt    Constant mesh spacing in t.
===== ==========================================================

Note that these routines uses PETSc, and can easily be extended to
parallel. There are different modules/packages that is used to solve
in parallel and serial, and need to be installed. Other than that, use
the '-np' argument of mpirun to choose between serial and parallel.
Using parallel, mpirun is not needed to run.
"""

# Import and initialize
import petsc4py, sys, time
import numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc


def solver_FE_MAT_PETSc(I, a, L, Nx, C, T):
	"""
	This solver uses matrix to solve with a Forward Euler scheme,
	using u = A*u_1, that is just a simple matrix-vector product.
	"""
	
	x = np.linspace(0, L, Nx+1)
	dx = x[1] - x[0]
	dt = C*dx**2/a
	Nt = int(round(T/float(dt)))
	t = np.linspace(0, T, Nt+1)
	t0 = time.clock()
	
	A = PETSc.Mat().createAIJ([Nx+1, Nx+1],nnz=3)
	for i in range(1, Nx+1):   # filling the non-zero entires
		A.setValue(i,i,1.-2*C)
		A.setValue(i-1,i,C)
		A.setValue(i,i-1,C)
	A.setValue(0,0,1); A.setValue(0,1,0);
	A.setValue(Nx,Nx,1); A.setValue(Nx,Nx-1,0);
	
	# Get the left and right hand side vectors with properties
	# from the matrix T, e.g. type and sizes.
	u, u_1 = A.getVecs()
	
	# Initialize first time step from the numpy array I
	u_1.setValues(range(Nx+1),I)
	
	# Assemble the vectors and matrix. This partition each part
	# on its correct process/rank
	A.assemble(); u.assemble(); u_1.assemble()
		
	# This is for outputting to file, to be read from another program.
	# Is more difficult if in parallel.
	if PETSc.Options().getBool('toFile',default=False):
		if PETSc.COMM_WORLD.getSize() > 1:
			if PETSc.COMM_WORLD.getRank() == 0:
				print 'Warning: not writing to file (using parallel)'
		else:
			W = PETSc.Viewer().createASCII('test3.txt',format=1)
			u_1.view(W)
	
	for n in range(0, Nt):
		# Solve for next step. This should be done without
		# matrix multiplication (solving the system directly).
		A.mult(u_1, u)
		
		# Copy new solution to the old vector
		u.copy(u_1)
		
		# Set boundary condition
		u_1.setValue(0,0); u_1.setValue(Nx,0)
		
		if PETSc.Options().getBool('toFile',default=False):
			u_1.view(W)
	return u_1, time.clock()-t0

def solver_FE_MATfree_PETSc(I, a, L, Nx, C, T):
	
	"""
	This solves the system using a Forward Euler scheme, but
	without using a explisit matrix. This calls advance-function
	to do the calculations at each time step. This saves memory (no
	need to store a matrix in the memory) and time (dont use any
	time setting up a matrix).
	
	For now we can call to advancer-functions (could be extended
	using Cython and f2py for speed):
	- advance_FE_looper: a plain scalar looper.
	- advance_FE_fastnumpy: a vectorized looper using Numpy. Note that
		in this function there is no hardcopying in the conversion
		between PETSc arrays and Numpy arrays, as they share the same
		memory location. This saves time during the calculations.
	"""
	
	x = np.linspace(0, L, Nx+1)
	dx = x[1] - x[0]
	dt = C*dx**2/a
	Nt = int(round(T/float(dt)))
	t = np.linspace(0, T, Nt+1)
	t0 = time.clock()
	
	# COMMENT
	da = PETSc.DA().create(sizes=[Nx+1], boundary_type=2,
		stencil_type=0, stencil_width=1)
	da.setUniformCoordinates(0, L)   # We never really use this in here
	u = da.createGlobalVector(); u_1 = da.createGlobalVector();
	A = da.createMatrix(); A.setType('aij')  # type is seqaij as default, must change this
	
	# Initialize init time step
	u_1.setValues(range(Nx+1),I)
	
	# Initialize the matrix A. Could maybe get this through DA, but do we want want that?
	# This means one creation call and one allocation call? What positive properties do we get?
	A = PETSc.Mat().createAIJ([Nx+1, Nx+1],nnz=3)
	for i in range(1, Nx+1):   # filling the non-zero entires
		A.setValue(i,i,1.-2*C)
		A.setValue(i-1,i,C)
		A.setValue(i,i-1,C)
	A.setValue(0,0,1); A.setValue(0,1,0);
	A.setValue(Nx,Nx,1); A.setValue(Nx,Nx-1,0);
	
	local_vec = da.createLocalVector(); local_vec_new = da.createLocalVector();
	
	# Assemble the vectors and the matrix. This distribute the objects out over the procs.
	A.assemble(); u.assemble(); u_1.assemble();
	
	# This can be cut out, this is for plotting in external
	# programs
	if PETSc.Options().getBool('toFile',default=False):
		if PETSc.COMM_WORLD.getSize() > 1:
			if PETSc.COMM_WORLD.getRank() == 0:
				print 'Warning: NOT writing to file (using parallel)'
				PETSc.Options().setValue('toFile',False)
		else:
			W = PETSc.Viewer().createASCII('test3.txt',format=1);
			u_1.view(W)
	
	
	for n in range(0, Nt):
		da.globalToLocal(u_1, local_vec)
		
		if PETSc.Options().getString('advance_method',default='fastnumpy') == 'looper':
			local_vec_new = advance_FE_looper(local_vec, C, da)
		else:
			local_vec_new = advance_FE_fastnumpy(local_vec, C, da)
		
		da.localToGlobal(local_vec_new, u_1)
	
		if PETSc.Options().getBool('draw',default=False):
			petsc_viz(da, u_1, 0.001)
	
	return u_1, time.clock()-t0

# Local functions for use in the MATfree function
def advance_FE_looper(u_1, C, da):
	u = da.createLocalVector()
	u_1.duplicate(u)
	
	for i in range(1,u_1.getSize()-1):
		val = C*u_1.getValue(i-1) + (1.-2*C)*u_1.getValue(i) + C*u_1.getValue(i+1)
		u.setValue(i, val)
	return u

def advance_FE_fastnumpy(u_1, C, da):
	u = u_1.getArray()

	u_advance = np.zeros_like(u)
	u_advance[1:-1] = C*u[:-2] + (1.-2*C)*u[1:-1] + C*u[2:]
	
	u_return = PETSc.Vec().createWithArray(u_advance)
	return u_return




def solver_BE_PETSc(I, a, L, Nx, C, T):
	
	"""
	This solve our system using an implicit Backward Euler scheme. There is no
	limitation on C, as in the explicit scheme (where C <= 0.5).
	
	We solve the system A*u = u_1 using a sparse A matrix.
	"""
	
	x = np.linspace(0, L, Nx+1)
	dx = x[1] - x[0]
	dt = C*dx**2/a
	Nt = int(round(T/float(dt)))
	t = np.linspace(0, T, Nt+1)
	t0 = time.clock()
	
	A = PETSc.Mat().createAIJ([Nx+1, Nx+1],nnz=3)
	for i in range(1, Nx+1):   # filling the diagonal and off-diagonal entries
		A.setValue(i,i,1.+2*C)
		A.setValue(i-1,i,-C)
		A.setValue(i,i-1,-C)
	# For boundary conditions
	A.setValue(0,0,1); A.setValue(0,1,0);
	A.setValue(Nx,Nx,1); A.setValue(Nx,Nx-1,0);
	
	# Create the to vectors, we need two of them (new and old). We're also
	# making a temporary RHS-array b
	u, u_1 = A.getVecs()
	
	# Assemble the matrix and vectors. This distribute the data on to the
	# different processors (among other things).
	A.assemblyBegin(); A.assemblyEnd()
	u.assemblyBegin(); u.assemblyEnd()
	u_1.assemblyBegin(); u_1.assemblyEnd()
	
	if PETSc.Options().getBool('toFile',default=False):
		if PETSc.COMM_WORLD.getSize() > 1:
			if PETSc.COMM_WORLD.getRank() == 0:
				print 'Warning: not writing to file (using parallel)'
		else:
			print 'Writing to file'
			W = PETSc.Viewer().createASCII('test3.txt',format=1)
			u_1.view(W)
	
	"""
	Setting up the KSP solver, the heart of PETSc. We use setFromOptions.
	
	The command line option for the KSP solver is -ksp_type <method> -pc_type <method>
	List of preconditioners (-pc_type):
	http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCType.html#PCType
	List of solvers (-ksp_type):
	http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPType.html#KSPType
	
	As an example: to use a direct solver method using LU, run the program using:
		-ksp_type preonly -pc_type lu
	To run with a Conjugate Gradient iterativ solver (nice for positive-definite and
	symmetric matrices often seen in Finite Difference systems) with incomplete
	Cholesky preconditioning, run with:
		-ksp_type cg -pc_type icc
	
	The solvers and preconditioners are also listed at (along with external packages):
	http://www.mcs.anl.gov/petsc/petsc-current/docs/linearsolvertable.html
	"""
	
	ksp = PETSc.KSP()
	ksp.create()
									# Setting a CG iterative solver as default, then one can override
									# this with command line options:
	ksp.setType('cg')				# KSP solver
	ksp.getPC().setType('icc')		# Preconditioner (LU factorization)
	
	ksp.setFromOptions()
	ksp.setOperators(A)				# Set which matrix to solve the problem with
	
	opt = PETSc.Options()
	if PETSc.COMM_WORLD.getRank() == 0:
		print 'For BE:\n   KSP type: ',ksp.getType(),'  PC type: ', ksp.getPC().getType(),'\n'
	
	for t in range(Nt):
		# Solve for next step using the solver set up above using
		# command-line options.
		ksp.solve(u_1,u)
		
		# Could we use maybe use this:   u, u_1 = u_1, u
		# to prevent hard-copying?
		u.copy(u_1)
		
		if PETSc.Options().getBool('toFile',default=False):
			u.view(W)
	
	return u_1, time.clock()-t0



def petsc_viz(da, u_1, sleeper=0.5):
	U = da.createNaturalVector()
	da.globalToNatural(u_1,U)
	scatter, U0 = PETSc.Scatter.toZero(U)
	scatter.scatter(U,U0,False,PETSc.Scatter.Mode.FORWARD)
	rank = PETSc.COMM_WORLD.getRank()
	solution = U.copy()
	draw = PETSc.Viewer.DRAW()
	draw(solution)
	time.sleep(sleeper)







