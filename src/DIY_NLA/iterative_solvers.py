import numpy as np
def conjugate_gradient(A, b, x0=None, max_iter=1000, tol=1e-10, callback=None):
    """
    Conjugate Gradient method for solving Ax = b
    
    Parameters:
    A : numpy.ndarray - Symmetric Positive Definite matrix
    b : numpy.ndarray - Right-hand side vector
    x0 : numpy.ndarray - Initial guess (optional)
    max_iter : int - Maximum number of iterations
    tol : float - Tolerance for residual norm
    callback : function - Optional callback for residual monitoring
    
    Returns:
    x : numpy.ndarray - Solution vector
    """
    # Initialize solution
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()
    
    # Initial residual
    r = b - A @ x
    p = r.copy()
    rsold = r.dot(r)
    
    # Iteration loop
    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / p.dot(Ap)
        x += alpha * p
        r -= alpha * Ap
        
        # Optional callback for residual monitoring
        if callback:
            callback(r)
            
        # Check convergence
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            break
            
        # Update search direction
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
        
    return x