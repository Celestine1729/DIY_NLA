# tests/test_direct_solvers.py
import numpy as np
from DNLAP.direct_solvers import solve_lu, solve_cholesky

def test_solve_lu():
    A = np.array([[3, 2], [1, 2]])
    b = np.array([5, 4])
    x = solve_lu(A, b)
    assert np.allclose(A @ x, b)

def test_solve_cholesky():
    A = np.array([[4, 1], [1, 3]])
    b = np.array([1, 2])
    x = solve_cholesky(A, b)
    assert np.allclose(A @ x, b)