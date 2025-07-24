# tests/test_iterative_solvers.py
import numpy as np
from numlinalg.iterative_solvers import conjugate_gradient

def test_conjugate_gradient():
    # Positive definite matrix
    A = np.array([[4, 1], [1, 3]])
    b = np.array([1, 2])
    x = conjugate_gradient(A, b)
    assert np.allclose(A @ x, b, atol=1e-6)