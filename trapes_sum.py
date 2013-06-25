# ------------------------------------------------------------------------
#
# Integration with the trapezoidal rule
# http://en.wikipedia.org/wiki/Trapezoidal_rule
# 
# I(f) = h*(0.5*f[0] + sum_of_f_array_without_endpoits + 0.5*f[-1]    (1)
#	Where
#		* h is the step length (b-a)/(N-1)
#		* f[0] is the start point and f[-1] is the last point
# 
# One test could be where f=2*x and the trapezoidal rule should calculate
# this integral exactly. The result should be f(b)**2 - f(a)**2.
# 
# ------------------------------------------------------------------------


# Import and initialize
import petsc4py, sys, numpy
petsc4py.init(sys.argv)
from petsc4py import PETSc

def trapes_rule(h, f):
	# Make a PETSc-vector of the vector f *without* the two endpoints
	# THIS createSeq DOES NOT ALLOW PARALLELISATION?
	#Pf = PETSc.Vec().createSeq(f.size-2)
	Pf = PETSc.Vec().createMPI(f.size-2,comm=PETSc.COMM_WORLD)
	Pf.setValues(range(f.size-2), f[1:-1])
	
	# Integrate using (1)
	integ = h*(Pf.sum() + 0.5*f[0] + 0.5*f[-1])
	
	return integ


# Start of script
if (len(sys.argv) > 1):
	n = int(sys.argv[1])
	if n == 1:
		print 'Error: n must be larger than 1'
		sys.exit()
else:
	n = 100


a = 0.
b = 1.
h = (b-a)/(n-1)
x = numpy.linspace(a,b,n)
#f = 2*x
f = 3.*x**2

this_integral = trapes_rule(h,f)
print this_integral