# tests/test_basic_ops.py
import numpy as np
import pytest
from DNLAP.basic_ops import dot_product, matrix_multiply, vector_norm, matrix_norm

def test_dot_product():
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    assert dot_product(v1, v2) == 32
    assert dot_product(v1, v1) == 14

def test_matrix_multiply():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    expected = np.array([[19, 22], [43, 50]])
    assert np.allclose(matrix_multiply(A, B), expected)

def test_vector_norm():
    v = np.array([3, 4])
    assert vector_norm(v) == 5.0
    assert vector_norm(v, p=1) == 7.0

def test_matrix_norm():
    A = np.array([[1, 2], [3, 4]])
    assert np.isclose(matrix_norm(A), np.linalg.norm(A))
    assert np.isclose(matrix_norm(A, p=np.inf), 7)