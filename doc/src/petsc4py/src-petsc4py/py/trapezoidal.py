import numpy as np

def trapezoidal_scalar(f, a, b, n):
    """
    Compute the integral of f from a to b with n intervals,
    using the Trapezoidal rule.
    """
    h = (b-a)/float(n)
    I = 0.5*(f(a) + f(b))
    for i in range(1, n):
        x = a + i*h
        I += f(x)
    I = h*I
    return I

def trapezoidal_vec(f, a, b, n):
    """
    Compute the integral of f from a to b with n intervals,
    using the Trapezoidal rule. Vectorized version.
    """
    x = np.linspace(a, b, n+1)
    f_vec = f(x)
    f_vec[0] /= 2.0
    f_vec[-1] /= 2.0
    h = (b-a)/float(n)
    I = h*np.sum(f_vec)
    return I

import nose.tools as nt

def test_trapezoidal(a=1.2, b=2.4, n=3):
    """Test that linear functions are exactly integrated."""
    f = lambda x: 4*x - 2.5
    import sympy as sm
    x = sm.Symbol('x')
    I_exact = sm.integrate(4*x - 2.5, (x, a, b))

    I_scalar = trapezoidal_scalar(f, a, b, n)
    I_vec    = trapezoidal_vec   (f, a, b, n)
    nt.assert_almost_equal(I_scalar, I_exact, places=10)
    nt.assert_almost_equal(I_vec,    I_exact, places=10)

if __name__ == '__main__':
    test_trapezoidal()

