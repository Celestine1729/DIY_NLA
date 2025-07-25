import numpy as np
from scipy.linalg import lu, cholesky, qr, svd

def lu_decomposition(A):
    P, L, U = lu(A)
    return P, L, U

def cholesky_decomposition(A):
    return cholesky(A, lower=True)

def qr_decomposition(A):
    return qr(A, mode='economic')

def svd_decomposition(A):
    return svd(A, full_matrices=False)