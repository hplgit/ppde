# Kjoring hos meg:     python2.6 backward_diffusion.py 1000 200

# ------------------------------------------------------------------------
# 
# 1-D diffusion equation using Backward Euler (implisit). The diffusion
# equation is
# 		du/dt = d2u/dx2.
# 
# We get a system A*un=unm1 where un is the new solution, unm1 is the
# solution at the last time step and A is a tridiagonal matrix.
# 
# This is unconditionally stable.
# 
# ------------------------------------------------------------------------

# Import and initialize
import petsc4py, sys, numpy
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Spatial points
if (len(sys.argv) > 1):
	n = int(sys.argv[1])
else:
	n = 100

# Temporal points (number of time steps)
if (len(sys.argv) > 2):
	Nt = int(sys.argv[2])
else:
	Nt = 100

if PETSc.COMM_WORLD.getRank() == 0:
	print 'Running with', n,'spatial points and',Nt,'time steps'


a_from = 0; b_to = 1; dx = (b_to-a_from)/(n-1.); dt = 100*dx**2
C = dt/(dx**2)    # C = dt/dx^2

# Need our tridiagonal matrix. That matrix are sparse by nature, so we
# use the default PETSc format AIJ which is sparse. One should (not
# must) preallocate the number of data points using nnz. For a
# tridiagonal matrix nnz=3 and for a diagonal matrix nnz=1. nnz is
# approx. number of elements per row.
T = PETSc.Mat().createAIJ([n, n],nnz=3)
for i in range(1, n):   # filling the diagonal and off-diagonal entries
	T.setValue(i,i,1.+2*C)
	T.setValue(i-1,i,-C)
	T.setValue(i,i-1,-C)
# For boundary conditions 0
T.setValue(0,0,1); T.setValue(0,1,0);
T.setValue(n-1,n-1,1); T.setValue(n-1,n-2,0);

# Create the two vectors, we need two of them (new and old).
un = PETSc.Vec().createSeq(n)
unm1 = PETSc.Vec().createSeq(n)

# The inital condition is an exponential, sine or linear (remember end points)
unm1.setValues(range(n),numpy.exp(-numpy.square(numpy.linspace(a_from,b_to,n)-0.5*(b_to-a_from))*50))
#unm1.setValues(range(n),numpy.sin(numpy.linspace(a_from,b_to,n)*2*numpy.pi))
#unm1.setValues(range(n),numpy.linspace(a_from,b_to,n))
unm1.setValue(0,0); unm1.setValue(n-1,0);

# Assemble all vectors and matrices.
un.assemblyBegin(); un.assemblyEnd()
unm1.assemblyBegin(); unm1.assemblyEnd()
T.assemblyBegin(); T.assemblyEnd()

'''
# Set up a solver. Using conjugate gradient (cg) for now (iterative),
# and incomplete Cholesky (icc) as preconditioner.
ksp = PETSc.KSP()
ksp.create()
ksp.setType('cg')             # Setting solver type
ksp.getPC().setType('icc')    # Preconditioner
ksp.setOperators(T)           # Set which matrix to solve the problem with
'''

# For a list of preconditioners:
# http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCType.html#PCType
# There seems to be convergence issues for this problem using an
# iterative solver (dependent of n). Using a direct solver solves without
# iteration and one does not get convergence issues. Direct methods tends
# to be slower(?)
#
# Here we uses LU factorization as a direct solver. This is set as a
# preconditioner, and we therefore sets the solver type to be 'preonly'.
ksp = PETSc.KSP()
ksp.create()
ksp.setType('preonly')        # Setting preonly as solver type
ksp.getPC().setType('lu')     # Preconditioner (LU factorization)
ksp.setOperators(T)           # Set which matrix to solve the problem with


plotting = False
# Importing plotting tool (correct use?)
if plotting:
	try:
		from matplotlib import pylab
		pylab.figure()
		x = numpy.linspace(-100,100,n)
	except ImportError:
		print 'WARNING: Matplotlib dont exist on this system, and', \
			'the results will not be plotted'
		plotting = False

outputToFile = False
if outputToFile:
	W = PETSc.Viewer().createASCII('test.txt',format=0)
	unm1.view(W)

# Now solve each time step sequentially
for t in range(Nt):
	# Run the solver. T is presetup and constant.
	ksp.solve(unm1,un)
	
	# Update unm1 to the new one, then we're ready for another timestep
	un.copy(unm1)
	
	# Boundary conditions
	unm1.setValue(0,0); unm1.setValue(n-1,0);
	
	# Is this the correct use? (Matlab style)
	if plotting:
		pylab.plot(x,un)
		pylab.show()
	
	if outputToFile:
		un.view(W)

if PETSc.COMM_WORLD.getRank() == 0:
	print 'Done with', Nt, 'timesteps'



