"""Interpolation facilities."""

import hashlib
import os
import numpy as np
import jax
import jax.numpy as jnp
from cosmologix.tools import get_cache_dir


def barycentric_weights(x):
    """Computes barycentric weights for interpolation points.

    Args:
        x (jnp.ndarray): The interpolation points.

    Returns:
        jnp.ndarray: The barycentric weights.
    """
    n = len(x)
    w = jnp.ones(n)

    # Compute weights using a numerically stable approach
    for j in range(n):
        # Product of (x_j - x_i) for i != j
        product = 1.0
        for i in range(n):
            if i != j:
                diff = x[j] - x[i]
                product *= diff
        # The weight is 1 / product to avoid overflow in the product
        w = w.at[j].set(1.0 / product if product != 0 else 1.0)

    return w


def chebyshev_nodes(n, a, b):
    """Computes n Chebyshev nodes of the second kind on the interval [a, b].

    Args:
        n (int): The number of nodes.
        a (float): The start of the interval.
        b (float): The end of the interval.

    Returns:
        jnp.ndarray: The Chebyshev nodes.
    """

    # Compute indices k = 0, 1, ..., n
    k = np.arange(n + 1)

    # Compute Chebyshev nodes on [-1, 1]
    x_cheb = np.cos(k * np.pi / n)  # jnp.cos((2 * k + 1) * jnp.pi / (2 * (n + 1)))

    # Map to [a, b]
    x_mapped = (b - a) / 2 * x_cheb + (a + b) / 2

    return jnp.array(x_mapped)


def barycentric_weights_chebyshev(n):
    """Computes barycentric weights for n+1 Chebyshev nodes.

    Args:
        n (int): The number of nodes.

    Returns:
        jnp.ndarray: The barycentric weights.
    """
    j = jnp.arange(n + 1)
    w = (-1.0) ** j
    w = w.at[0].set(w[0] / 2.0)
    w = w.at[n].set(w[n] / 2.0)
    return w


def barycentric_interp(x_tab, y_tab, x_query, w=None):
    """Performs barycentric interpolation.

    This is reputed to be more stable numerically than Newton's formulae but
    can cause issues regarding differentiability.

    Args:
        x_tab (jnp.ndarray): The x-coordinates of the tabulated points.
        y_tab (jnp.ndarray): The y-coordinates of the tabulated points.
        x_query (jnp.ndarray): The x-coordinate at which to interpolate.
        w (jnp.ndarray, optional): The barycentric weights. If None, they are
            computed. Defaults to None.

    Returns:
        jnp.ndarray: The interpolated value.
    """
    if w is None:
        w = barycentric_weights(x_tab)

    xq = jnp.atleast_1d(x_query)
    exact_matches = x_tab == xq[0]
    exact_match = (exact_matches.any()).astype(int)
    exact_idx = exact_matches.argmax()

    def exact_case():
        return y_tab[exact_idx]

    def interp_case():
        # Compute numerator and denominator of barycentric formula
        diffs = xq[0] - x_tab
        # Avoid division by zero by setting a large weight for exact matches
        terms = w * y_tab / diffs
        num = jnp.sum(terms)
        den = jnp.sum(w / diffs)
        return num / den

    return jax.lax.switch(exact_match, [interp_case, exact_case])


def newton_divided_differences(x, y):
    """Computes the divided differences for Newton's interpolation.

    Args:
        x (jnp.ndarray): The x-coordinates of the points.
        y (jnp.ndarray): The y-coordinates of the points.

    Returns:
        jnp.ndarray: The divided differences.
    """
    n = len(x)
    # Initialize the divided difference table with y values
    coeffs = jnp.zeros((n, n))
    coeffs = coeffs.at[:, 0].set(y)

    # Compute divided differences
    for j in range(1, n):
        for i in range(n - j):
            coeffs = coeffs.at[i, j].set(
                (coeffs[i + 1, j - 1] - coeffs[i, j - 1]) / (x[i + j] - x[i])
            )

    # Return the coefficients (first row of the table)
    return coeffs[0, :]


def cached_newton_divided_differences(x, func, cache_dir=None):
    """Computes or retrieves cached Newton divided differences.

    This wrapper caches the result of `newton_divided_differences(x, func(x))`
    to disk using a unique filename based on the inputs. If the cache exists,
    it loads and returns the result directly.

    Args:
        x (jnp.ndarray): Array of x-coordinates (interpolation points).
        func (callable): Function that takes x as input and returns y-values.
        cache_dir (str, optional): Directory where cache files are stored.
            Defaults to None.

    Returns:
        jnp.ndarray: Array of divided difference coefficients.

    Notes:
        The cache filename is generated from a hash of x and `func.__name__`.
        Cache files are stored as .npy files.
        The cache directory is created if it doesn’t exist.
    """

    if cache_dir is None:
        cache_dir = get_cache_dir()

    # Generate a unique cache key based on x and func name
    x_hash = hashlib.sha256(x.tobytes()).hexdigest()[:16]  # Shorten for readability
    func_name = func.__name__
    cache_filename = f"newton_diff_{func_name}_{x_hash}.npy"
    cache_path = os.path.join(cache_dir, cache_filename)

    # Create cache directory if it doesn’t exist
    os.makedirs(cache_dir, exist_ok=True)

    # Check if cached result exists
    if os.path.exists(cache_path):
        # Load and return cached coefficients
        coeffs = jnp.asarray(np.load(cache_path))
        return coeffs

    x = jnp.asarray(x)
    y = func(x)

    # Compute coefficients if not cached
    coeffs = newton_divided_differences(x, y)

    # Save to cache (convert to NumPy for .npy compatibility)
    np.save(cache_path, np.asarray(coeffs))

    return coeffs


def newton_interp(x_tab, y_tab, coeffs=None):
    """Evaluates Newton's interpolation polynomial.

    Args:
        x_tab (jnp.ndarray): The x-coordinates of the tabulated points.
        y_tab (jnp.ndarray): The y-coordinates of the tabulated points.
        coeffs (jnp.ndarray, optional): The divided differences. If None,
            they are computed. Defaults to None.

    Returns:
        callable: The interpolation function.
    """
    if coeffs is None:
        coeffs = newton_divided_differences(x_tab, y_tab)

    n = len(x_tab)

    # @jax.jit
    def eval_horner(xq):
        def body_fun(i, val):
            return jnp.multiply(val, xq - x_tab[n - i]) + coeffs[n - i]

        result = jnp.full(xq.shape, coeffs[-1])
        result = jax.lax.fori_loop(2, n + 1, body_fun, result)
        return result

    return eval_horner


def linear_interpolation(
    x: jnp.ndarray, y_bins: jnp.ndarray, x_bins: jnp.ndarray
) -> jnp.ndarray:
    """Performs linear interpolation between set points.

    Args:
        x (jnp.ndarray): x-coordinates for interpolation.
        y_bins (jnp.ndarray): y-coordinates of the set points.
        x_bins (jnp.ndarray): x-coordinates of the set points.

    Returns:
        jnp.ndarray: Interpolated y-values.
    """
    bin_index = jnp.digitize(x, x_bins) - 1
    w = (x - x_bins[bin_index]) / (x_bins[bin_index + 1] - x_bins[bin_index])
    return (1 - w) * y_bins[bin_index] + w * y_bins[bin_index + 1]


def newton_interpolant(func, a, b, n=10):
    """Returns a polynomial interpolant of a function.

    The polynomial is evaluated using the Newton formula whose precomputation
    is in O(n^2).

    Args:
        func (callable): The function to interpolate.
        a (float): The start of the interval.
        b (float): The end of the interval.
        n (int, optional): The number of Chebyshev nodes to use. Defaults to 10.

    Returns:
        callable: The interpolation function.
    """
    nodes = chebyshev_nodes(n, a, b)
    return newton_interp(nodes, func(nodes))


def barycentric_interpolant(func, a, b, n=10):
    """Returns a polynomial interpolant of a function.

    The polynomial is evaluated using the barycentric formula, which is faster
    to precompute for Chebyshev nodes than the Newton formula and should be
    more stable numerically. However, the JAX implementation is not fully
    differentiable.

    Args:
        func (callable): The function to interpolate.
        a (float): The start of the interval.
        b (float): The end of the interval.
        n (int, optional): The number of Chebyshev nodes to use. Defaults to 10.

    Returns:
        callable: The interpolation function.
    """
    nodes = chebyshev_nodes(n, a, b)
    weights = barycentric_weights_chebyshev(n)
    return jax.vmap(
        lambda x: barycentric_interp(nodes, func(nodes), x, weights), in_axes=(0,)
    )


def linear_interpolant(func, a, b, n=10):
    """Returns a linear interpolant of a function.

    Args:
        func (callable): The function to interpolate.
        a (float): The start of the interval.
        b (float): The end of the interval.
        n (int, optional): The number of regularly spaced nodes to use.
            Defaults to 10.

    Returns:
        callable: The interpolation function.
    """
    nodes = jnp.linspace(a, b * (1 + 1e-6), n)
    return lambda x: linear_interpolation(x, func(nodes), nodes)
