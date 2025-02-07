'''The module provides two second order methods to solve non-linear
problems
'''
import jax
import jax.numpy as jnp
import time

def flatten_vector(v):
    ''' Transforms a vector with a pytree structure into a standard array
    '''
    return jnp.hstack([jnp.ravel(v[p]) for p in v])

def unflatten_vector(p, v):
    ''' Give a standard array v the exact same pytree structure as p
    '''
    st = {}
    i = 0
    for k in p:
        j = i + jnp.size(p[k])
        st[k] = jnp.reshape(v[i:j], jnp.shape(p[k]))
        i = j
    return st

#def conjugate_gradient(Ax, b, x0=None, tol=1e-6, max_iter=10):
#    """
#    Solves the linear system Ax = b using the Conjugate Gradient method.
#
#    Parameters:
#    - Ax: Function that computes Ax where A is implicitly defined through this function.
#    - b: Right-hand side vector.
#    - x0: Initial guess for the solution. If None, starts with zeros.
#    - tol: Tolerance for convergence; when the norm of residual is below this, we stop.
#    - max_iter: Maximum number of iterations.
#
#    Returns:
#    - x: Approximate solution to Ax = b
#    - info: Dictionary containing 'iter': number of iterations, 'res': final residual norm
#    """
#
#    b = jnp.asarray(b)
#    n = b.shape[0]
#
#    # If no initial guess is provided, start with zero vector
#    if x0 is None:
#        x = jnp.zeros_like(b)
#    else:
#        x = jnp.asarray(x0)
#
#    # Compute initial residual
#    r = b - Ax(x)
#    p = r.copy()  # Initial search direction
#    r_norm = jnp.linalg.norm(r)
#    
#    for i in range(max_iter):
#        if r_norm < tol:  # Check for convergence
#            break
#
#        Ap = Ax(p)
#        alpha = r_norm ** 2 / jnp.dot(p, Ap)
#        x = x + alpha * p
#        r = r - alpha * Ap
#        
#        r_new_norm = jnp.linalg.norm(r)
#        beta = r_new_norm ** 2 / r_norm ** 2
#        p = r + beta * p
#        
#        r_norm = r_new_norm
#
#    return x, {'iter': i, 'res': r_norm}
#
#def hessian_vector_product(g, x, v):
#    return jax.jvp(g, (x,), (v,))[1]
#
#def newton_cg(func, x0, gradient, hvp, niter=1000, tol=1e-3, nitercg=10):
#    xi = flatten_vector(x0)
#    loss = [func(xi)]
#    tstart = time.time()
#    timings = [0]
#    def cgn_step(xi):
#        return conjugate_gradient(hvp(xi), gradient(xi), max_iter=nitercg)[0]
#        #cgn(hvp(xi), gradient(xi), 1000)[0]
#    for i in range(niter):
#        xi -= cgn_step(xi)
#        loss.append(func(xi))
#        timings.append(time.time()-tstart)
#        if loss[-2] - loss[-1] < tol:
#            break
#    timings = jnp.array(timings)
#    return unflatten_vector(x0, xi), {'loss': loss, 'timings': timings}

def newton(func, x0, g=None, H=None, niter=1000, tol=1e-3):
    xi = flatten_vector(x0)
    loss = lambda x:func(unflatten_vector(x0, x))
    losses = [loss(xi)]
    tstart = time.time()
    if g is None:
        g = jax.grad(loss)
    if H is None:
        H = jax.hessian(loss)
    timings = [0]
    for i in range(niter):
        xi -= jnp.linalg.solve(H(xi), g(xi))
        losses.append(loss(xi))
        timings.append(time.time()-tstart)
        if losses[-2] - losses[-1] < tol:
            break
    timings = jnp.array(timings)
    return unflatten_vector(x0, xi), {'loss': losses, 'timings': timings}
