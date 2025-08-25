"""
Test unit for pysolve module
"""

import pytest
import numpy as np
from numroot import NonlinearSolver

TOL = 1E-3

def f_1(_x):
    return (_x-1)*(_x+3)

def f_2(_x):
    return (_x-3)*(_x+4)

def f_3(_x):
    return np.log(_x)

def f_4(_x):
    return 3*np.exp(-2*_x)-1

def f_5(_x):
    return np.sin(_x)/_x

def df_1(_x):
    return 2*_x+2

def df_2(_x):
    return 2*_x+1

def df_3(_x):
    return 1/_x

def df_4(_x):
    return -6*np.exp(-2*_x)

def df_5(_x):
    return (np.sin(_x)-_x*np.cos(_x))/_x**2

solver = NonlinearSolver()

@pytest.mark.parametrize("f, x_a, x_b", \
    [
        (f_1, 0, 2),
        (f_2, -1, 5),
        (f_3, 2, 0.5),
        (f_4, 0, 1),
        (f_5, 0.1, 4),
    ])
def test_bisect(f, x_a, x_b):
    res, _ = solver.bisect(f, x_a, x_b)
    assert f(res) < TOL

@pytest.mark.parametrize("f, x_0, x_1", \
    [
        (f_1, 0, 2),
        (f_2, -1, 5),
        (f_3, 2, 0.5),
        (f_4, 0, 1),
        (f_5, 0.1, 4),
    ])
def test_secant(f, x_0, x_1):
    res, _ = solver.secant(f, x_0, x_1)
    assert f(res) < TOL

@pytest.mark.parametrize("f, df, x_0", \
    [
        (f_1, df_1, 0),
        (f_2, df_2, -1),
        (f_3, df_3, 0.5),
        (f_4, df_4, -5),
        (f_4, df_4, 0),
        (f_5, df_5, 0.3),
    ])
def test_newton_ralphson(f, df, x_0):
    res, _ = solver.newton_ralphson(f, df, x_0)
    assert f(res) < TOL
