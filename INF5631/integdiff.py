# Import and initialize
import petsc4py, sys, time
import numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc

def trapezoidal_PETSc(f, a, b, n):
	"""
    Compute the integral of f from a to b with n intervals,
    using the Trapezoidal rule. PETSc version, which one could
	be used in parallel.
    """
	x = np.linspace(a, b, n+1)
	h = (b-a)/float(n)
	t0 = time.clock()
	f_vec = f#(x)
	f_vec[0] /= 2.0
	f_vec[-1] /= 2.0
	
	# Create a distributed array, assemble (doing the
	# distribution), get the local ownership span, and then fill
	# the ranks localpart of it.
	Pf = PETSc.Vec().createMPI(f_vec.size,comm=PETSc.COMM_WORLD)
	Pf.assemble()
	[Istart, Iend] = Pf.getOwnershipRange()
	Pf.setValues(range(Istart,Iend),f_vec[Istart:Iend])
	
	# Now we find the total sum (which apperently also
	# get scattered to all ranks). Note that in the start
	# if this routine we divided the end points by 2.
	tot_sum = h*Pf.sum()
	
	# Every rank has now a correct sum, and each ranks
	# version can be use. This is unlike the
	# differentiate function, where only root rank is correct.
	return tot_sum, time.clock()-t0


def differentiate_PETSc(f, a, b, n):
	"""
	Compute the discrete derivative of a Python function
	f on [a,b] using n intervals. Internal points apply
	a centered difference, while end points apply a one-sided
	difference. PETSc serial/parallel version.
	"""
	
	x = np.linspace(a, b, n+1)  # mesh
	f_vec = f#(x)
	dx = x[1] - x[0]
	t0 = time.clock()
	
	# Only root rank need to allocate a large array for storage.
	# The otherranks need still to return something, so we set
	# df to a zero.
	if PETSc.COMM_WORLD.getRank() == 0:
		df = np.zeros_like(x)
	else:
		df = np.zeros(1)
	
	# We need one array containing the left part of the vector,
	# and one for the right part. This is distributed during the
	# assembly.
	diffL = PETSc.Vec().createMPI(f_vec.size-2)
	diffR = PETSc.Vec().createMPI(f_vec.size-2)
	
	# Each rank owns its local part, given by getOwnershipRange.
	# From this we copy each part of f_vec, and then assemble.
	# As diffL and diffR is the same, their ownership ranges must
	# be the same.
	[Istart, Iend] = diffR.getOwnershipRange()
	diffL.setValues(range(Istart,Iend),f_vec[Istart:Iend])
	diffR.setValues(range(Istart,Iend),f_vec[Istart+2:Iend+2])
	diffL.assemble(); diffR.assemble()
	
	# Now we do the calculations. In pure vector numpy code, this
	# would be:    df[1:-1] = (f_vec[2:] - f_vec[:-2])/(2*dx)
	diffR.axpby(-1,1,diffL)
	diffR.scale(1/(2*dx))
	
	# Now that each rank has calculcated each local part of the
	# differentiated vector, we need to gather it all to root rank
	# with the PETSc Scatter module.
	scatter, diff_ = PETSc.Scatter.toZero(diffR)
	scatter.scatter(diffR,diff_,False,PETSc.Scatter.Mode.FORWARD)
	
	# Only root can collect the PETSc array to the Numpy array df.
	if PETSc.COMM_WORLD.getRank() == 0:
		df[1:-1] = diff_.getArray()
		
		# End points
		df[0]  = (f_vec[1]  - f_vec[0]) /dx
		df[-1] = (f_vec[-1] - f_vec[-2])/dx
	
	# Note that all other ranks then rank=0 returns a 0,
	# that is, pure junk. Timing is returned for benchmarking.
	return df, time.clock()-t0
	

def differentiate_vec(f, a, b, n):
	"""
	Compute the discrete derivative of a Python function
	f on [a,b] using n intervals. Internal points apply
	a centered difference, while end points apply a one-sided
	difference. Vectorized version.
	"""
	x = np.linspace(a, b, n+1)  # mesh
	df = np.zeros_like(x)       # df/dx
	f_vec = f#(x)
	dx = x[1] - x[0]
	t0 = time.clock()
	
	# Internal mesh points
	df[1:-1] = (f_vec[2:] - f_vec[:-2])/(2*dx)
	# End points
	df[0]  = (f_vec[1]  - f_vec[0]) /dx
	df[-1] = (f_vec[-1] - f_vec[-2])/dx
	
	return df, time.clock()-t0

