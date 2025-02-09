"""The module provides two second order methods to solve non-linear
problems
"""

import jax
import jax.numpy as jnp
import time


def flatten_vector(v):
    """Transforms a vector with a pytree structure into a standard array"""
    return jnp.hstack([jnp.ravel(v[p]) for p in v])


def unflatten_vector(p, v):
    """Give a standard array v the exact same pytree structure as p"""
    st = {}
    i = 0
    for k in p:
        j = i + jnp.size(p[k])
        st[k] = jnp.reshape(v[i:j], jnp.shape(p[k]))
        i = j
    return st

def newton(func, x0, g=None, H=None, niter=1000, tol=1e-3):
    xi = flatten_vector(x0)
    loss = lambda x: func(unflatten_vector(x0, x))
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
        timings.append(time.time() - tstart)
        if losses[-2] - losses[-1] < tol:
            break
    timings = jnp.array(timings)
    return unflatten_vector(x0, xi), {"loss": losses, "timings": timings}

def gauss_newton_partial(wres, jac, x0, fixed, niter=1000, tol=1e-3,full=False):
    """
    Perform partial Gauss-Newton optimization for non-linear least squares problems.

    This function implements the Gauss-Newton method with partial updates, where some 
    parameters are fixed during optimization. It iteratively minimizes the sum of 
    squared residuals by approximating the Hessian matrix.

    Parameters:
    - wres (callable): Function to compute weighted residuals. Takes (x, fixed) as arguments.
      - x: Current parameter values (free parameters).
      - fixed: Fixed parameters that do not change during optimization.
    - jac (callable): Function to compute the Jacobian of `wres`. Takes (x, fixed) as arguments.
      - x: Current parameter values.
      - fixed: Fixed parameters.
    - x0 (array-like): Initial guess for the free parameters.
    - fixed (array-like): Fixed parameters that are not optimized.
    - niter (int): Maximum number of iterations to perform. Default is 1000.
    - tol (float): Tolerance for convergence based on the change in loss. Default is 1e-3.
    - full (bool): If True, includes the Fisher Information Matrix (FIM) in the output. Default is False.

    Returns:
    - x (array-like): Optimized values of the free parameters.
    - extra (dict): Additional information about the optimization process:
      - 'loss' (list): Losses (sum of squared residuals) at each iteration.
      - 'timings' (list): Time taken at each iteration in seconds.
      - 'FIM' (array-like, optional): Fisher Information Matrix if `full` is True.

    Notes:
    - The function uses the Gauss-Newton method, which assumes that the Hessian of 
      the sum of squares can be approximated by J^T*J, where J is the Jacobian.
    - Convergence is determined when the decrease in loss between iterations is 
      less than `tol`.
    - This method is particularly useful for parameter estimation in non-linear 
      least squares problems where some parameters are known or fixed.

    Raises:
    - May raise a LinAlgError if the system of equations is singular or nearly singular,
      causing problems with `jnp.linalg.solve`.

    Example:
    >>> def residuals(x, fixed): return x - fixed
    >>> def jacobian(x, fixed): return jnp.ones_like(x)
    >>> result, info = gauss_newton_partial(residuals, jacobian, jnp.array([2.0]), jnp.array([1.0]), niter=10, tol=1e-6)
    """
    timings = [time.time()]
    x = x0
    losses=[]
    for i in range(niter):
        R = wres(x, fixed)
        losses.append((R**2).sum())
        if i > 1:
            if (losses[-2] - losses[-1] < tol):
                break
        J = jac(x, fixed)
        g = J.T@R
        dx = jnp.linalg.solve(J.T@J, g)
        x = x - dx
        timings.append(time.time())
    extra = {'loss':losses, 'timings':timings}
    if full:
        extra['FIM'] = jnp.linalg.inv(J.T@J)
    return x, extra

def newton_partial(loss, x0, g, H, fixed, niter=1000, tol=1e-3):
    xi = x0
    losses = [loss(xi, fixed)]
    tstart = time.time()
    timings = [0]
    for i in range(niter):
        xi -= jnp.linalg.solve(H(xi, fixed), g(xi, fixed))
        losses.append(loss(xi, fixed))
        timings.append(time.time() - tstart)
        if losses[-2] - losses[-1] < tol:
            break
    timings = jnp.array(timings)
    return xi, {"loss": losses, "timings": timings}
