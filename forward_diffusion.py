# ------------------------------------------------------------------------
# 
# Based on backward_diffusion.py. We employ a Forward Euler (explicit)
# scheme on the 1-D diffusion equation, given by
# 		du/dt = d2u/dx2.
#
# The initial condition is given under, and the boundary condition is 0
# at both ends.
#
# Here the system solves un = T*unm1, where un is the new system, T is
# tridiagonal matrix given under, and unm1 is the vector from the last
# time step. Remember that this explicit scheme is limited by the
# Courant condition: given a spatial step dx, the C=dt/dx**2 can not be
# larger than some value. This sets a limit on how big a time step
# one can use. Violation of the Courant condition results in an unstable
# system, that tends to go to infinity.
# 
# This implementation does not use the KSP-objects, solving a system of
# linear equations. Since this is a pure explicit system, we only need
# to multiply the matrix with the last step to get our new vector.
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

	
print 'Running with', n,'spatial points and',Nt,'time steps'

a_from = 0; b_to = 1; dx = (b_to-a_from)/(n-1.);
C = 0.01*dx**2/(dx**2)    # C = dt/dx^2, remember the Courant condition

# Need the tridiagonal matrix of the right hand side. That matrix are
# sparse by nature, so we use the default PETSc format AIJ which is
# sparse. One should (not must) preallocate the number of data points
# using nnz. For a tridiagonal matrix nnz=3. nnz is # approx. number
# of elements per row.
T = PETSc.Mat().createAIJ([n, n],nnz=3)
for i in range(1, n):   # filling the diagonal and off-diagonal entries
	T.setValue(i,i,1.-2*C)
	T.setValue(i-1,i,C)
	T.setValue(i,i-1,C)
# For boundary conditions
T.setValue(0,0,1); T.setValue(0,1,0);
T.setValue(n-1,n-1,1); T.setValue(n-1,n-2,0);

# Create the two vectors, we need two of them (new and old).
un = PETSc.Vec().createMPI(n); un.set(1);
unm1 = PETSc.Vec().createMPI(n)

# The inital condition is an exponential or sine for now
unm1.setValues(range(n),numpy.exp(-numpy.square(numpy.linspace(a_from,b_to,n)-0.5*(b_to-a_from))*50))
#unm1.setValues(range(n),numpy.sin(numpy.linspace(a_from,b_to,n)*2*numpy.pi))
#unm1.setValues(range(n),numpy.linspace(a_from,b_to,n))
unm1.setValue(0,0); unm1.setValue(n-1,0);


# Assemble all vectors and matrices.
un.assemblyBegin()
un.assemblyEnd()
unm1.assemblyBegin()
unm1.assemblyEnd()
T.assemblyBegin()
T.assemblyEnd()


plotting = False
# Importing plotting tool
if plotting:
	try:
		from matplotlib import pylab
		pylab.figure()
		x = numpy.linspace(-100,100,n)
	except ImportError:
		print 'WARNING: Matplotlib dont exist on this system, and', \
			'the results will not be plotted'
		plotting = False

# Writing output to file.
outputToFile = True
if outputToFile:
	W = PETSc.Viewer().createASCII('test.txt',format=1)
	unm1.view(W)


# Now solve each time step sequentially
for t in range(Nt):
	# Multiply last vector with the tridiagonal vector
	T.mult(unm1, un)
	
	# Update unm1 to the new one, then we're ready for another timestep
	un.copy(unm1)
	unm1.setValue(0,0); unm1.setValue(n-1,0);
	
	# Is this the correct use? (Matlab style)
	if plotting:
		pylab.plot(x,un)
		pylab.show()
	
	if outputToFile:
		un.view(W)

print 'Done with', Nt, 'timesteps'