"""The module provides two second order methods to solve non-linear
problems
"""

import time
from typing import Callable, Dict, Any
import jax
import jax.numpy as jnp
from .parameters import Planck18


def flatten_vector(v):
    """Flatten a vector with a pytree structure into a 1D array.

    This function takes a dictionary representing a pytree (a nested structure of arrays)
    and concatenates all arrays into a single 1D array by raveling each component.

    Parameters
    ----------
    v : dict
        A dictionary where each value is a JAX array or a nested structure that can be
        flattened into an array.

    Returns
    -------
    jnp.ndarray
        A 1D JAX array containing the flattened and concatenated values from the input.

    Examples
    --------
    >>> v = {'a': jnp.array([1, 2]), 'b': jnp.array([[3, 4], [5, 6]])}
    >>> flatten_vector(v)
    DeviceArray([1, 2, 3, 4, 5, 6], dtype=int32)
    """
    return jnp.hstack([jnp.ravel(v[p]) for p in v])


def unflatten_vector(p, v):
    """Reconstruct a pytree structure from a flattened array using a template.

    This function takes a flattened 1D array and reshapes it to match the structure
    of the template dictionary `p`, restoring the original shapes of the arrays.

    Parameters
    ----------
    p : dict
        A dictionary serving as a template, where each value is a JAX array whose shape
        defines the structure to be restored.
    v : jnp.ndarray
        A 1D JAX array containing the flattened values to be reshaped.

    Returns
    -------
    dict
        A dictionary with the same keys as `p` and values reshaped to match the
        corresponding array shapes in `p`.

    Examples
    --------
    >>> p = {'a': jnp.array([0, 0]), 'b': jnp.array([[0, 0], [0, 0]])}
    >>> v = jnp.array([1, 2, 3, 4, 5, 6])
    >>> unflatten_vector(p, v)
    {'a': DeviceArray([1, 2], dtype=int32),
     'b': DeviceArray([[3, 4], [5, 6]], dtype=int32)}
    """
    st = {}
    i = 0
    for k in p:
        j = i + jnp.size(p[k])
        st[k] = jnp.reshape(v[i:j], jnp.shape(p[k]))
        i = j
    return st


def restrict(f: Callable, fixed_params: dict = None) -> Callable:
    """Modify a function by fixing some of its parameters.

    This is similar to functools.partial but allows fixing parts of
    the first pytree argument.

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
    If mu expects a dictionary with 'Omega_bc' and 'w',
    restrict(mu, {'w': -1}) returns a function of 'Omega_bc' only.

    """
    if fixed_params is None:
        fixed_params = {}

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
    varied_dict = {}
    for p in varied:
        fixed.pop(p)
        varied_dict[p] = complete[p]
    if flat:
        return lambda x: func(dict(unflatten_vector(varied, x), fixed)), varied_dict
    return lambda x: func(dict(x, **fixed)), varied_dict


def partial(func: Callable, param_subset: Dict[str, Any]) -> Callable:
    """Create a new function that operates on a subset of parameters.

    This function adapts an input function to accept a flattened array representing
    a subset of parameters, while fixing the remaining parameters from a provided point.

    Parameters
    ----------
    func : callable
        The original function that takes a dictionary of parameters and returns a value.
    param_subset : dict
        A dictionary specifying the structure of the parameter subset to be optimized.

    Returns
    -------
    callable
        A new function that takes a flattened array `x` (representing the subset of
        parameters) and a dictionary `point` (containing fixed parameters), and returns
        the result of `func` applied to the combined parameters.

    Examples
    --------
    >>> def func(params): return params['a'] + params['b']
    >>> param_subset = {'a': jnp.array([0])}
    >>> new_func = partial(func, param_subset)
    >>> point = {'b': jnp.array([10])}
    >>> new_func(jnp.array([5]), point)
    DeviceArray([15], dtype=int32)
    """

    def _func(x, point):
        return func(dict(unflatten_vector(param_subset, x), **point))

    return _func


class UnconstrainedParameterError(Exception):
    """Raised when a parameter is unconstrained in the fit."""

    def __init__(self, unconstrained_params):
        self.params = unconstrained_params
        message = "Unconstrained parameters detected:\n" + "\n".join(
            f"  {name}: σ = {unc:.2f}" for name, unc in unconstrained_params
        )
        super().__init__(message)


class DegenerateParametersError(Exception):
    """Raised when perfect degeneracy between parameters is detected."""

    def __init__(self, degeneracies):
        self.params = degeneracies
        message = "Unconstrained parameters detected:\n" + "\n".join(
            f"  {param1} <-> {param2}: correlation = {corr_val:.4f}"
            for param1, param2, corr_val in degeneracies
        )
        super().__init__(message)


def analyze_fim_for_unconstrained(fim, param_names):
    """Analyze FIM for unconstrained parameters and degeneracies."""
    # Check for unconstrained parameters (zero entries in the FIM)
    threshold = 1e-10  # Arbitrary large value for "unconstrained"
    unconstrained = [
        (name, float(unc))
        for name, unc in zip(param_names, jnp.diag(fim))
        if unc < threshold
    ]
    if unconstrained:
        print("\nUnconstrained Parameters:")
        for name, unc in unconstrained:
            print(f"  {name}: FIM = {unc:.2f} (effectively unconstrained)")
    return unconstrained


def analyze_fim_for_degeneracies(fim, param_names):
    """Analyze FIM for degeneracies between parameters."""
    # Compute covariance matrix
    cov = jnp.linalg.inv(fim)
    variances = jnp.diag(cov)
    uncertainties = jnp.sqrt(variances)

    # Compute correlation matrix
    corr = cov / jnp.outer(uncertainties, uncertainties)
    corr = jnp.where(jnp.isnan(corr), 0, corr)  # Handle NaN from division by zero

    # Check for perfect degeneracies (|corr| ≈ 1, excluding diagonal)
    degeneracy_threshold = 0.999  # Close to ±1
    degeneracies = []
    for i, name in enumerate(param_names):
        for j in range(i + 1, len(param_names)):
            if abs(corr[i, j]) > degeneracy_threshold:
                degeneracies.append((name, param_names[j], float(corr[i, j])))

    # if degeneracies:
    #    print("\nPerfect Degeneracies Detected (|correlation| > 0.999):")
    #    for param1, param2, corr_val in degeneracies:
    #        print(f"  {param1} <-> {param2}: correlation = {corr_val:.4f}")
    # else:
    #    print("\nNo perfect degeneracies detected.")
    return degeneracies


class LikelihoodSum:

    def __init__(self, likelihoods):
        self.likelihoods = likelihoods

    def negative_log_likelihood(self, params):
        return jnp.sum(
            jnp.array([l.negative_log_likelihood(params) for l in self.likelihoods])
        )

    def weighted_residuals(self, params):
        return jnp.hstack([l.weighted_residuals(params) for l in self.likelihoods])

    def initial_guess(self, params):
        for l in self.likelihoods:
            params = l.initial_guess(params)
        return params

    def draw(self, params):
        for l in self.likelihoods:
            l.draw(params)


def gauss_newton_prep(
    func: Callable, params_subset: Dict[str, Any]
) -> tuple[Callable, Callable]:
    """Prepare a function and its Jacobian for the Gauss-Newton algorithm.

    This function creates a restricted version of the input function that operates on
    a subset of parameters and computes its Jacobian using JAX's forward-mode
    automatic differentiation. The result is suitable for use in Gauss-Newton optimization.

    Parameters
    ----------
    func : callable
        The original function that takes a dictionary of parameters and returns a value.
        Typically, this is a residual or cost function.
    params_subset : dict
        A dictionary specifying the structure of the parameter subset to be optimized.

    Returns
    -------
    tuple[callable, callable]
        A tuple containing:
        - The restricted function, JIT-compiled, that operates on a flattened array of
          the parameter subset and fixed parameters.
        - The Jacobian of the restricted function, JIT-compiled, computed with respect
          to the flattened parameter subset.

    Notes
    -----
    The returned functions are JIT-compiled for performance using `jax.jit`. The Jacobian
    is computed using forward-mode automatic differentiation (`jax.jacfwd`).

    Examples
    --------
    >>> def func(params): return params['a'] ** 2 + params['b']
    >>> params_subset = {'a': jnp.array([0.])}
    >>> f, jac = gauss_newton_prep(func, params_subset)
    >>> x = jnp.array([2.])
    >>> point = {'b': jnp.array([3.])}
    >>> f(x, point)
    DeviceArray(7., dtype=float32)
    >>> jac(x, point)
    DeviceArray([[4.]], dtype=float32)
    """
    f = jax.jit(partial(func, params_subset))
    return f, jax.jit(jax.jacfwd(f))


def fit(likelihoods, fixed=None, verbose=False, initial_guess=None):
    """Fit a set of likelihoods using the Gauss-Newton method with
    partial parameter fixing.

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
        - 'bestfit': The best-fit parameters as a dictionary matching
          the initial guess format.
        - 'FIM': An approximation of the Fisher Information Matrix
          (FIM) at the best fit.
        - 'loss': The progression of loss values during optimization
          (from `gauss_newton_partial`).
        - 'timings': The time taken for each iteration of the
          optimization (from `gauss_newton_partial`).

    Notes:
    - The function uses `LikelihoodSum` to combine multiple
      likelihoods into one, which must be a class that can call
      `.initial_guess()` with `Planck18` for a starting point.

    The optimization process involves:

    1. Determining an initial guess from the combined likelihoods,
    updating with fixed parameters.
    2. Preparing the weighted residuals and Jacobian for optimization.
    3. Using a partial Gauss-Newton method for minimization, where
    only non-fixed parameters are optimized.
    4. Computing the Fisher Information Matrix for the best fit,
    providing insight into parameter uncertainties.

    Example:
    >>> priors = [likelihoods.Planck2018Prior(), likelihoods.DES5yr()]
    >>> fixed = {'Omega_k':0., 'm_nu':0.06, 'Neff':3.046, 'Tcmb': 2.7255}
    >>> result = fit(priors, fixed=fixed)
    >>> print(result['bestfit'])

    """
    if fixed is None:
        fixed = {}

    if initial_guess is None:
        initial_guess = Planck18.copy()
    likelihood = LikelihoodSum(likelihoods)

    # Pick up a good starting point
    params = likelihood.initial_guess(initial_guess.copy())
    initial_guess = params.copy()
    for p in fixed:
        assert p in params, "Unknow parameter name {p}"
        initial_guess.pop(p)
    params.update(fixed)

    # Restrict the function to free parameters and jit compilation
    wres, wjac = gauss_newton_prep(likelihood.weighted_residuals, initial_guess)

    # Prep the fit starting point
    x0 = flatten_vector(initial_guess)
    if verbose:
        print(initial_guess)

    # Quick inspection to look for degeracies
    jac = wjac(x0, fixed)  # pylint: disable=not-callable
    fim = jac.T @ jac
    unconstrained = analyze_fim_for_unconstrained(fim, list(initial_guess.keys()))
    if unconstrained:
        raise UnconstrainedParameterError(unconstrained)
    degenerate = analyze_fim_for_degeneracies(fim, list(initial_guess.keys()))
    if degenerate:
        raise DegenerateParametersError(degenerate)
    # Minimization
    xbest, extra = gauss_newton_partial(wres, wjac, x0, fixed, verbose=verbose)

    # report the residuals at the end of the fit
    extra["residuals"] = wres(xbest, fixed)  # pylint: disable=not-callable

    # Compute approximation of the FIM
    jac = wjac(xbest, fixed)  # pylint: disable=not-callable
    inverse_fim = jnp.linalg.inv(jac.T @ jac)
    extra["inverse_FIM"] = inverse_fim

    # Unflatten the vectors for conveniency
    extra["x"] = xbest
    extra["bestfit"] = unflatten_vector(initial_guess, xbest)

    return extra


# def newton(func, x0, g=None, H=None, niter=50, tol=1e-3):
#    xi = flatten_vector(x0)
#    loss = lambda x: func(unflatten_vector(x0, x))
#    losses = [loss(xi)]
#    tstart = time.time()
#    if g is None:
#        g = jax.jit(jax.grad(loss))
#    if H is None:
#        H = jax.jit(jax.hessian(loss))
#    print(x0)
#    h = H(xi)
#    print(h)
#    G = g(xi)
#    print(G)
#    print(jnp.linalg.solve(h, G))
#    timings = [0]
#    for i in range(niter):
#        print(f"{i}/{niter}")
#        xi -= jnp.linalg.solve(H(xi), g(xi))
#        print(xi)
#        losses.append(loss(xi))
#        timings.append(time.time() - tstart)
#        if losses[-2] - losses[-1] < tol:
#            break
#    timings = jnp.array(timings)
#    return unflatten_vector(x0, xi), {"loss": losses, "timings": timings}


# pylint: disable=invalid-name
def gauss_newton_partial(
    wres, jac, x0, fixed, niter=50, tol=1e-3, full=False, verbose=False
):
    """Perform partial Gauss-Newton optimization for non-linear least squares problems.

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
    - full (bool): If True, includes the Fisher Information Matrix
      (FIM) in the output. Default is False.

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
    >>> result, info = gauss_newton_partial(residuals, jacobian,
                                            jnp.array([2.0]),
                                            jnp.array([1.0]), niter=10, tol=1e-6)
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
        if verbose:
            print(x)
            print(dx)
        x = x - dx
        timings.append(time.time())
    extra = {"loss": losses, "timings": timings}
    if full:
        extra["FIM"] = jnp.linalg.inv(J.T @ J)
    return x, extra


# def newton_partial(loss, x0, g, H, fixed, niter=1000, tol=1e-3):
#    xi = x0
#    losses = [loss(xi, fixed)]
#    tstart = time.time()
#    timings = [0]
#    for i in range(niter):
#        xi -= jnp.linalg.solve(H(xi, fixed), g(xi, fixed))
#        losses.append(loss(xi, fixed))
#        timings.append(time.time() - tstart)
#        if losses[-2] - losses[-1] < tol:
#            break
#    timings = jnp.array(timings)
#    return xi, {"loss": losses, "timings": timings}
