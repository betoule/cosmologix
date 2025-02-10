"""The module provides two second order methods to solve non-linear
problems
"""

import jax
import jax.numpy as jnp
import time
from typing import Callable
from .likelihoods import LikelihoodSum, Planck18


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


def restrict(f: Callable, fixed_params: dict = {}) -> Callable:
    """
    Modify a function by fixing some of its parameters.

    This is similar to functools.partial but allows fixing parts of the first pytree argument.

    Parameters:
    -----------
    f: Callable
        A function with signature f(params, *args, **keys) where params is a pytree.
    fixed_params: dict
        Parameters to fix with provided values.

    Returns:
    --------
    Callable
        Function with same signature but with parameters fixed to their provided values.

    Example:
    --------
    If mu expects a dictionary with 'Omega_m' and 'w',
    restrict(mu, {'w': -1}) returns a function of 'Omega_m' only.
    """

    def g(params, *args, **kwargs):
        updated_params = fixed_params.copy()
        updated_params.update(params)
        return f(updated_params, *args, **kwargs)

    return g


def restrict_to(func, complete, varied, flat=True):
    """Create a new function by restricting the input parameters of `func` to a subset.

    This utility function allows you to fix some parameters of `func` while allowing
    others to vary. It effectively turns a function with multiple parameters into one
    where only a subset of those parameters can be changed, with the others fixed.

    Parameters:
    - func (callable): The original function to be modified. It should accept a dictionary
      of parameters as its argument.
    - complete (dict): A dictionary containing all parameters that `func` could accept,
      with their values set to what should be used when not varied.
    - varied (list or tuple): A list of parameter names that should be allowed to vary.
    - flat (bool): If True, the input to the returned lambda will be expected as a
      flat vector (list or array) which will be converted into the dictionary form
      for `func`. If False, the input should already be a dictionary containing
      the varied parameters. Default is True.

    Returns:
    - callable: A lambda function that either:
        - If `flat` is True, takes a flat vector of values for the `varied` parameters
          and returns the result of calling `func` with those values and the fixed
          parameters combined.
        - If `flat` is False, takes a dictionary with keys matching `varied`, merges
          it with `fixed`, and calls `func` with this merged dictionary.

    Notes:
    - This function is particularly useful in optimization routines where you need to
      hold some parameters constant while optimizing others.
    - See also `restrict` for another way to restrict the function by
      specifying only the parameter to fix.

    Example:
    >>> def original_func(params):
    ...     return params['a'] + params['b'] * params['c']
    >>> restricted_func = restrict_to(original_func, {'a': 1, 'b': 2, 'c': 3}, ['b', 'c'])
    >>> restricted_func([2, 3])  # 'a' is fixed at 1, 'b' and 'c' are varied
    7

    """
    fixed = complete.copy()
    V = {}
    for p in varied:
        fixed.pop(p)
        V[p] = complete[p]
    if flat:
        return lambda x: func(dict(unflatten_vector(varied, x), fixed)), V
    else:
        return lambda x: func(dict(x, **fixed)), V


def partial(func, param_subset):
    def _func(x, point):
        return func(dict(unflatten_vector(param_subset, x), **point))

    return _func


def newton_prep(func, params_subset):
    f = jax.jit(partial(func, params_subset))
    return f, jax.jit(jax.grad(f)), jax.jit(jax.hessian(f))


def gauss_newton_prep(func, params_subset):
    f = partial(func, params_subset)
    return f, jax.jit(jax.jacfwd(f))


def fit(likelihoods, fixed={}):
    """Fit a set of likelihoods using the Gauss-Newton method with partial parameter fixing.

    This function combines multiple likelihoods, optimizes the
    parameters using an initial guess possibly augmented by fixed
    parameters, and then applies the Gauss-Newton optimization method.

    Parameters:
    - likelihoods: A list of likelihood object, each expected to
      provide a weighted_residuals function of parameters as a
      dictionary and return weighted residuals or similar metrics.
    - fixed (dict): A dictionary of parameters to be fixed during the optimization
      process. Keys are parameter names, values are their fixed values. Default is an
      empty dictionary.

    Returns:
    - dict: A dictionary containing:
        - 'x': The optimized parameter values in a flattened form.
        - 'bestfit': The best-fit parameters as a dictionary matching the initial guess format.
        - 'FIM': An approximation of the Fisher Information Matrix (FIM) at the best fit.
        - 'loss': The progression of loss values during optimization (from `gauss_newton_partial`).
        - 'timings': The time taken for each iteration of the optimization (from `gauss_newton_partial`).

    Notes:
    - The function uses `LikelihoodSum` to combine multiple likelihoods into one,
      which must be a class that can call `.initial_guess()` with `Planck18` for a starting point.

    The optimization process involves:
    1. Determining an initial guess from the combined likelihoods, updating with fixed parameters.
    2. Preparing the weighted residuals and Jacobian for optimization.
    3. Using a partial Gauss-Newton method for minimization, where only non-fixed parameters are optimized.
    4. Computing the Fisher Information Matrix for the best fit, providing insight into parameter uncertainties.

    Example:
    >>> priors = [likelihoods.Planck2018Prior(), likelihoods.DES5yr()]
    >>> fixed = {'Omega_k':0., 'm_nu':0.06, 'Neff':3.046, 'Tcmb': 2.7255}
    >>> result = fit(priors, fixed=fixed)
    >>> print(result['bestfit'])
    """
    likelihood = LikelihoodSum(likelihoods)

    # Pick up a good starting point
    params = likelihood.initial_guess(Planck18)
    initial_guess = params.copy()
    for p in fixed:
        assert p in params, "Unknow parameter name {p}"
        initial_guess.pop(p)
    params.update(fixed)
    # Restrict the function to free parameters and jit compilation
    wres, wjac = gauss_newton_prep(likelihood.weighted_residuals, initial_guess)

    # Minimization
    x0 = flatten_vector(initial_guess)
    xbest, extra = gauss_newton_partial(wres, wjac, x0, fixed)

    # Compute approximation of the FIM
    J = wjac(xbest, fixed)
    FIM = jnp.linalg.inv(J.T @ J)
    extra["FIM"] = FIM

    # Unflatten the vectors for conveniency
    extra["x"] = xbest
    extra["bestfit"] = unflatten_vector(initial_guess, xbest)

    return extra


def newton(func, x0, g=None, H=None, niter=50, tol=1e-3):
    xi = flatten_vector(x0)
    loss = lambda x: func(unflatten_vector(x0, x))
    losses = [loss(xi)]
    tstart = time.time()
    if g is None:
        g = jax.jit(jax.grad(loss))
    if H is None:
        H = jax.jit(jax.hessian(loss))
    print(x0)
    h = H(xi)
    print(h)
    G = g(xi)
    print(G)
    print(jnp.linalg.solve(h, G))
    timings = [0]
    for i in range(niter):
        print(f"{i}/{niter}")
        xi -= jnp.linalg.solve(H(xi), g(xi))
        print(xi)
        losses.append(loss(xi))
        timings.append(time.time() - tstart)
        if losses[-2] - losses[-1] < tol:
            break
    timings = jnp.array(timings)
    return unflatten_vector(x0, xi), {"loss": losses, "timings": timings}


def gauss_newton_partial(wres, jac, x0, fixed, niter=50, tol=1e-3, full=False):
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
    losses = []
    for i in range(niter):
        R = wres(x, fixed)
        losses.append((R**2).sum())
        if i > 1:
            if losses[-2] - losses[-1] < tol:
                break
        J = jac(x, fixed)
        g = J.T @ R
        dx = jnp.linalg.solve(J.T @ J, g)
        x = x - dx
        timings.append(time.time())
    extra = {"loss": losses, "timings": timings}
    if full:
        extra["FIM"] = jnp.linalg.inv(J.T @ J)
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
