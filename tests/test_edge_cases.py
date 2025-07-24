# tests/test_edge_cases.py
import numpy as np
import pytest
from numlinalg import *

def test_singular_matrix():
    A = np.array([[1, 1], [1, 1]])
    b = np.array([2, 2])
    with pytest.raises(np.linalg.LinAlgError):
        solve_lu(A, b)

def test_non_spd_cholesky():
    A = np.array([[1, 2], [3, 4]])  # Not positive definite
    with pytest.raises(np.linalg.LinAlgError):
        cholesky_decomposition(A)

def test_empty_matrix():
    with pytest.raises(ValueError):
        vector_norm(np.array([]))