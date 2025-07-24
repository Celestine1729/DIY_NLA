def conjugate_gradient(A, b, max_iter=1000, tol=1e-10):
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    for _ in range(max_iter):
        alpha = np.dot(r, r) / np.dot(p, A @ p)
        x += alpha * p
        r_new = r - alpha * (A @ p)
        if np.linalg.norm(r_new) < tol:
            break
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
    return x