def power_iteration(A, max_iter=1000, tol=1e-10):
    v = np.random.rand(A.shape[0])
    for _ in range(max_iter):
        Av = A @ v
        v_new = Av / np.linalg.norm(Av)
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new
    eigenvalue = v.T @ A @ v
    return eigenvalue, v