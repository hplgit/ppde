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
	f_vec = f#f(x)
	f_vec[0] /= 2.0
	f_vec[-1] /= 2.0
	h = (b-a)/float(n)
	
	Pf = PETSc.Vec().createMPI(f_vec.size,comm=PETSc.COMM_WORLD)
	Pf.assemble()
	Pf.setValues(range(f_vec.size),f_vec)
	
	return h*Pf.sum()

def differentiate_PETSc_notDA(f, a, b, n):
	"""
	Compute the discrete derivative of a Python function
	f on [a,b] using n intervals. Internal points apply
	a centered difference, while end points apply a one-sided
	difference. PETSc serial/parallel version.
	"""
	
	x = np.linspace(a, b, n+1)  # mesh
	df = np.zeros_like(x)       # df/dx
	f_vec = f#(x)
	dx = x[1] - x[0]
	
	diff_l = PETSc.Vec().createMPI(f_vec.size-1)
	diff_r = PETSc.Vec().createMPI(f_vec.size-1)
	
	diff_l.setValues(range(f_vec.size-1),f[:-1]); diff_r.setValues(range(f_vec.size-1),f[1:])
	
	diff_l.assemble(); diff_r.assemble()
	# Here we do the actual differentiating
	diff_r.axpby(-1,1,diff_l)
	
	# Scale the end points with 1/dx and the others with 1/(2*dx)
	diff_r.scale(1/(2*dx))
	diff_r.setValue(0,2*diff_r.getValue(0))
	diff_r.setValue(diff_r.getSize()-1,2*diff_r.getValue(diff_r.getSize()-1))
	
	np_vec = diff_r.getArray()
	return np_vec
#

def differentiate_PETSc(f, a, b, n):
	"""
	Compute the discrete derivative of a Python function
	f on [a,b] using n intervals. Internal points apply
	a centered difference, while end points apply a one-sided
	difference. PETSc serial/parallel version.
	"""
	
	x = np.linspace(a, b, n+1)  # mesh
	df = np.zeros_like(x)       # df/dx
	f_vec = f#(x)
	dx = x[1] - x[0]
	
	da = PETSc.DA().create(sizes=[f_vec.size-1], boundary_type=2,
		stencil_type=0, stencil_width=1)
	
	diff_l = da.createGlobalVector(); diff_r = da.createGlobalVector()
	diff_l.setValues(range(f_vec.size-1),f_vec[:-1]); diff_r.setValues(range(f_vec.size-1),f_vec[1:])
	diff_l.assemble(); diff_r.assemble()
	
	diff_l_local = da.createLocalVector(); diff_r_local = da.createLocalVector()
	da.globalToLocal(diff_l, diff_l_local); da.globalToLocal(diff_r, diff_r_local)
	
	diff_r_local.axpby(-1,1,diff_l_local)
	
	diff_r_local.scale(1/dx)
	
	da.localToGlobal(diff_r_local, diff_r)
	
	# U will be the resulting vector (spanning everything, not just local)
	U = da.createNaturalVector()
	da.globalToNatural(diff_r,U)
	scatter, U0 = PETSc.Scatter.toZero(U)
	scatter.scatter(U,U0,False,PETSc.Scatter.Mode.FORWARD)
	
	# The following makes of course U0 only corrent for root rank
	if PETSc.COMM_WORLD.getRank() == 0:
		U0.setValue(0,2*U0.getValue(0))
		U0.setValue(U0.getSize()-1,2*U0.getValue(U0.getSize()-1))
	
	return U0.getArray()
	
	

def differentiate_vec(f, a, b, n):
	"""
	Compute the discrete derivative of a Python function
	f on [a,b] using n intervals. Internal points apply
	a centered difference, while end points apply a one-sided
	difference. Vectorized version.
	"""
	
	x = np.linspace(a, b, n+1)
	#df = np.zeros(n)
	f_vec = f#(x)
	dx = x[1] - x[0]
	df = f_vec/dx
	return df
#








