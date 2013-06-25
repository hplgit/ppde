# ------------------------------------------------------------------------
# 
# 2-D diffusion equation using Backward Euler (implicit). This is based
# upon the 1-D solver. The A*un=b system is explained here:
# http://pauli.uni-muenster.de/tp/fileadmin/lehre/NumMethoden/WS0910/ScriptPDE/Heat.pdf
# 
# Some of the code is inspired by:
# https://fs.hlrs.de/projects/par/par_prog_ws/pdf/petsc_exa_heat.pdf
#
# For simplicity the boundaries is zero everywhere. In that case the
# right-hand-side b is simply unm1 (the solution at the last time step).
#
# To run this in parallel, use:
# mpirun -np <number of processes> python backward_diffusion_2d.py
#     (with optional arguments given below: <nx> <ny> <Nt>
# This can be run in serial with -np 1 or just run with python, ignoring
# mpirun. Note that running this in parallel requires additional
# packages in PETSc.
# 
# This scheme is unconditionally stable.
# 
# ------------------------------------------------------------------------

# Import and initialize
import petsc4py, sys, numpy, math
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Read spatial points in first axis
if (len(sys.argv) > 1):
	nx = int(sys.argv[1])
else:
	nx = 100
	
# Spatial points in second axis
if (len(sys.argv) > 2):
	ny = int(sys.argv[2])
else:
	ny = 100

# Temporal points (number of time steps)
if (len(sys.argv) > 3):
	Nt = int(sys.argv[3])
else:
	Nt = 100

if PETSc.COMM_WORLD.getRank() == 0:
	print 'Running with nx='+str(nx)+' by ny='+str(ny)+' ('+str(nx*ny)+') spatial points and '+str(Nt)+' time steps'

# Using unit square, and calculating dx/dy and Cx=dt/dx**2.
xfrom = 0; xto = 1; dx = (xto-xfrom)/(nx-1.); Cx = 0.1*dx**2/(dx**2)
yfrom = 0; yto = 1; dy = (yto-yfrom)/(ny-1.); Cy = 0.1*dy**2/(dy**2)

# We now create, allocate and set the matrix A. We allocate 5 data
# elements per row. As this is spread out over the processes, we
# only need to set some of the elements. The span is given by
# getOwnershipRange(), (start, end).
#
# A is a five-band matrix. See the pdf refered to over. Note that
# the bands over and under the diagonal has "gaps" in them.
A = PETSc.Mat().createAIJ([nx*ny, nx*ny],nnz=5)
[Istart, Iend] = A.getOwnershipRange()
for I in range(Istart, Iend):
	i=I/nx; j=I-i*nx;
	
	if (i>0):
		J = I-nx
		A.setValue(I,J,-Cx)
	if (i<ny-1):
		J = I+nx
		A.setValue(I,J,-Cx)
	if (j>0):
		J = I-1
		A.setValue(I,J,-Cy)
	if (j<nx-1):
		J = I+1
		A.setValue(I,J,-Cy)
	A.setValue(I,I,1+2*Cx+2*Cy)

# We now create the vectors. These are vectorial form of the solution
# matrices (new and old time step). They are stored row by row(?).
un = PETSc.Vec().createMPI(nx*ny)     # The new time step
unm1 = PETSc.Vec().createMPI(nx*ny)   # The old time step

# Now we initialize time step zero as a exponential "spike" in the
# middle of the unit square.
for i in range(nx):
	for j in range(ny):
		unm1.setValue(i*ny+j,math.exp(-((i*dx-0.5)**2 + (j*dy-0.5)**2)*100))

# All boundary values equals zero
for i in range(nx):
	unm1.setValue(i*ny,0); unm1.setValue(i*ny+ny-1,0);
for j in range(ny):
	unm1.setValue(j,0);	unm1.setValue((nx-1)*ny+j,0);
		
# We need to assemble the matrix and vectors.
A.assemblyBegin(); A.assemblyEnd()
un.assemblyBegin(); un.assemblyEnd()
unm1.assemblyBegin(); unm1.assemblyEnd()

# Most parallel solver (both direct and iterative) requires external
# packages. For instance, LU factorization as a preconditioner using
# mpiaij (parallel AIJ) requires either PaStiX, SuperLU_Dist or MUMPS.
# Note also that SuperLU (for serial) and SuperLU_Dist (for parallel)
# is not the same packages.
ksp = PETSc.KSP()
ksp.create()
ksp.setType('preonly')        # Setting preonly as solver type
ksp.getPC().setType('lu')     # Preconditioner (LU factorization)
ksp.setOperators(A)

# Set to True if you want to write solution to file. This might
# be used for further analysis or plotting.
outputToFile = False
if outputToFile:
	W = PETSc.Viewer().createASCII('test2d.txt',format=0)
	unm1.view(W)

# We can now finaly solve for each time step.
for t in range(Nt):
	# Solve using the ksp object initialized over. RHS is unm1.
	ksp.solve(unm1,un)
	
	# Copy new solution to the unm1 as storage for next the
	# next solving.
	un.copy(unm1)
	
	# Again, setting the boundary to zero.
	for i in range(nx):
		unm1.setValue(i*ny,0); unm1.setValue(i*ny+ny-1,0);
	for j in range(ny):
		unm1.setValue(j,0);	unm1.setValue((nx-1)*ny+j,0);
	
	# Write next solution to file if outputToFile is True.
	if outputToFile:
		unm1.view(W)

# Done!