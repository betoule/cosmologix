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
    k = jnp.arange(n+1)
    
    # Compute Chebyshev nodes on [-1, 1]
    x_cheb = jnp.cos(k * jnp.pi/n)#jnp.cos((2 * k + 1) * jnp.pi / (2 * (n + 1)))
    
    # Map to [a, b]
    x_mapped = (b - a) / 2 * x_cheb + (a + b) / 2
    
    return x_mapped

def barycentric_weights_chebyshev(n):
    """Compute barycentric weights for n+1 Chebyshev nodes."""
    j = jnp.arange(n + 1)
    w = (-1.) ** j
    w = w.at[0].set(w[0]/2.)
    w = w.at[n].set(w[n]/2.)
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
            coeffs = coeffs.at[i, j].set((coeffs[i + 1, j - 1] - coeffs[i, j - 1]) / (x[i + j] - x[i]))
    
    # Return the coefficients (first row of the table)
    return coeffs[0, :]

def newton_interp(x_tab, y_tab, x_query, coeffs=None):
    """Evaluate Newton's interpolation polynomial at x_query."""
    if coeffs is None:
        coeffs = newton_divided_differences(x_tab, y_tab)
    
    n = len(x_tab)
    
    def eval_single(xq):
        # Initialize result with the first coefficient
        result = coeffs[0]
        # Compute the product term (x - x_0)(x - x_1)...
        product = 1.0
        for i in range(1, n):
            product *= (xq - x_tab[i - 1])
            result += coeffs[i] * product
        return result
    
    return eval_single(x_query)
