import numpy as np
from src.DIY_NLA.decompositions import lu_decomposition , cholesky_decomposition

def solve_lu(A, b):
    P, L, U = lu_decomposition(A)
    y = np.linalg.solve(L, P @ b)
    return np.linalg.solve(U, y)

def solve_cholesky(A, b):
    L = cholesky_decomposition(A)
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)