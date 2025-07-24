# tests/test_eigen.py
import numpy as np
from src.DIY_NLA.eigen import power_iteration

def test_power_iteration():
    A = np.array([[2, 1], [1, 2]])
    eigval, eigvec = power_iteration(A)
    # Validate eigen-equation A*v = λ*v
    assert np.allclose(A @ eigvec, eigval * eigvec, atol=1e-6)