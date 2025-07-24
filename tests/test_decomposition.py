# tests/test_decompositions.py
import numpy as np
from src.DIY_NLA.decompositions import lu_decomposition, cholesky_decomposition, qr_decomposition, svd_decomposition

def test_lu_decomposition():
    A = np.array([[4, 3], [6, 3]])
    P, L, U = lu_decomposition(A)
    assert np.allclose(P @ L @ U, A)

def test_cholesky_decomposition():
    A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
    L = cholesky_decomposition(A)
    assert np.allclose(L @ L.T, A)

def test_qr_decomposition():
    A = np.random.rand(3, 2)
    Q, R = qr_decomposition(A)
    assert np.allclose(Q @ R, A)
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]))  # Orthogonality

def test_svd_decomposition():
    A = np.array([[1, 2], [3, 4], [5, 6]])
    U, s, Vt = svd_decomposition(A)
    assert np.allclose(U @ np.diag(s) @ Vt, A)