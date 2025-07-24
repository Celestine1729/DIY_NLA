import numpy as np

def dot_product(v1, v2):
    return np.dot(v1, v2)

def matrix_multiply(A, B):
    return A @ B

def vector_norm(v, p=2):
    return np.linalg.norm(v, ord=p)

def matrix_norm(A, p='fro'):
    return np.linalg.norm(A, ord=p)