import jax
import jax.numpy as jnp


def barycentric_weights(x):
    """Compute barycentric weights for interpolation points x."""
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
    """Compute n Chebyshev nodes of the second kind on the interval [a, b]."""
    # Compute indices k = 0, 1, ..., n
    k = jnp.arange(n + 1)

    # Compute Chebyshev nodes on [-1, 1]
    x_cheb = jnp.cos(k * jnp.pi / n)  # jnp.cos((2 * k + 1) * jnp.pi / (2 * (n + 1)))

    # Map to [a, b]
    x_mapped = (b - a) / 2 * x_cheb + (a + b) / 2

    return x_mapped


def barycentric_weights_chebyshev(n):
    """Compute barycentric weights for n+1 Chebyshev nodes."""
    j = jnp.arange(n + 1)
    w = (-1.0) ** j
    w = w.at[0].set(w[0] / 2.0)
    w = w.at[n].set(w[n] / 2.0)
    return w


def barycentric_interp(x_tab, y_tab, x_query, w=None):
    """Perform barycentric interpolation at x_query given tabulated points (x_tab, y_tab).

    This is reputed to be more stable numerically than Newton's
    formulae but can causes issues regarding to differentiability.
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
    """Compute the divided differences for Newton's interpolation."""
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


def newton_interp(x_tab, y_tab, coeffs=None):
    """Evaluate Newton's interpolation polynomial."""
    if coeffs is None:
        coeffs = newton_divided_differences(x_tab, y_tab)

    n = len(x_tab)

    @jax.jit
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
    """
    Perform linear interpolation between set points.

    Parameters:
    -----------
    x: jnp.ndarray
        x coordinates for interpolation.
    y_bins, x_bins: jnp.ndarray
        y and x coordinates of the set points.

    Returns:
    --------
    jnp.ndarray: Interpolated y values.
    """
    bin_index = jnp.digitize(x, x_bins) - 1
    w = (x - x_bins[bin_index]) / (x_bins[bin_index + 1] - x_bins[bin_index])
    return (1 - w) * y_bins[bin_index] + w * y_bins[bin_index + 1]


def newton_interpolant(func, a, b, n=10):
    """Return a polynomial interpolant of the provided function on interval [a, b] using n chebyshev_nodes

    The polynomial is evaluated using the Newton formula whose precomputation is in O(n²).
    """
    nodes = chebyshev_nodes(n, a, b)
    # coeffs = newton_divided_differences(nodes, func(nodes))
    # return lambda x: newton_interp(nodes, None, x, coeffs=coeffs)
    return newton_interp(nodes, func(nodes))


def barycentric_interpolant(func, a, b, n=10):
    """Return a polynomial interpolant of the provided function on interval [a, b] using n chebyshev nodes.

    The polynomial is evaluated using the barycentric formula which is
    faster to precompute for chebyshev nodes than the newton formula
    and should be more stable numerically. However the jax
    implementation is not fully differentiable.
    """
    nodes = chebyshev_nodes(n, a, b)
    weights = barycentric_weights_chebyshev(n)
    return jax.vmap(
        lambda x: barycentric_interp(nodes, func(nodes), x, weights), in_axes=(0,)
    )


def linear_interpolant(func, a, b, n=10):
    """Return a linear interpolant of the provided function on interval [a, b] using n regularly spaced nodes."""
    nodes = jnp.linspace(a, b * (1 + 1e-6), n)
    return lambda x: linear_interpolation(x, func(nodes), nodes)
