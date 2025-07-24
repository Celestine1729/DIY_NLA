# tests/test_iterative_solvers.py
import numpy as np
import pytest
from src.DIY_NLA.iterative_solvers import conjugate_gradient



def test_conjugate_gradient_larger():
    # Generate larger SPD matrix
    np.random.seed(42)
    B = np.random.rand(10, 10)
    A = B.T @ B + np.eye(10)  # Make SPD
    x_true = np.random.rand(10)
    b = A @ x_true
    
    # Solve with CG
    x = conjugate_gradient(A, b, max_iter=100, tol=1e-8)
    
    # Verify solution accuracy
    assert np.allclose(A @ x, b, atol=1e-6)
    assert np.allclose(x, x_true, atol=1e-4)

