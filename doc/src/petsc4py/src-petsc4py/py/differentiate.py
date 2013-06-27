import numpy as np

def differentiate_scalar(f, a, b, n):
    """
    Compute the discrete derivative of a Python function
    f on [a,b] using n intervals. Internal points apply
    a centered difference, while end points apply a one-sided
    difference.
    """
    x = np.linspace(a, b, n+1)  # mesh
    df = np.zeros_like(x)       # df/dx
    f_vec = f(x)
    dx = x[1] - x[0]
    # Internal mesh points
    for i in range(1, n):
        df[i] = (f_vec[i+1] - f_vec[i-1])/(2*dx)
    # End points
    df[0]  = (f_vec[1]  - f_vec[0]) /dx
    df[-1] = (f_vec[-1] - f_vec[-2])/dx
    return df

def differentiate_vec(f, a, b, n):
    """
    Compute the discrete derivative of a Python function
    f on [a,b] using n intervals. Internal points apply
    a centered difference, while end points apply a one-sided
    difference. Vectorized version.
    """
    x = np.linspace(a, b, n+1)  # mesh
    df = np.zeros_like(x)       # df/dx
    f_vec = f(x)
    dx = x[1] - x[0]
    # Internal mesh points
    df[1:-1] = (f_vec[2:] - f_vec[:-2])/(2*dx)
    # End points
    df[0]  = (f_vec[1]  - f_vec[0]) /dx
    df[-1] = (f_vec[-1] - f_vec[-2])/dx
    return df

import nose.tools as nt

def test_differentiate(a=1.2, b=2.4, n=3):
    """Test that linear functions are exactly differentiated."""
    f = lambda x: 4*x - 2.5
    df_scalar = differentiate_scalar(f, a, b, n)
    df_vec    = differentiate_vec   (f, a, b, n)
    df_exact = np.zeros_like(df_scalar) + 4
    df_scalar_diff = np.abs(df_scalar - df_exact).max()
    df_vec_diff    = np.abs(df_vec    - df_exact).max()
    nt.assert_almost_equal(df_scalar_diff, 0, places=10)
    nt.assert_almost_equal(df_vec_diff,    0, places=10)

if __name__ == '__main__':
    test_differentiate()

