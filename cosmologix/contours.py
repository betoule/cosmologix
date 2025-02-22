from cosmologix.likelihoods import LikelihoodSum
from cosmologix.fitter import (
    restrict_to,
    restrict,
    newton,
    partial,
    flatten_vector,
    newton_partial,
    gauss_newton_partial,
    gauss_newton_prep,
    unflatten_vector,
)
from cosmologix.tools import conflevel_to_delta_chi2
from cosmologix import Planck18
import jax.numpy as jnp
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm


def frequentist_contour_2D(
    likelihoods,
    grid={"Omega_m": [0.18, 0.48, 30], "w": [-0.6, -1.5, 30]},
    varied=[],
    fixed=None,
):
    likelihood = LikelihoodSum(likelihoods)

    # Update the initial guess with the nuisance parameters associated
    # with all involved likelihoods
    params = likelihood.initial_guess(Planck18)
    if fixed is not None:
        params.update(fixed)
        wres = restrict(likelihood.weighted_residuals, fixed)
        initial_guess = params.copy()
        for p in fixed:
            initial_guess.pop(p)
    else:
        wres, initial_guess = restrict_to(
            likelihood.weighted_residuals,
            params,
            varied=list(grid.keys()) + varied,
            flat=False,
        )
    # Looking for the global minimum
    wres_, J = gauss_newton_prep(wres, initial_guess)
    x0 = flatten_vector(initial_guess)
    xbest, extra = gauss_newton_partial(wres_, J, x0, {})
    bestfit = unflatten_vector(initial_guess, xbest)

    # Exploring the chi2 space
    explored_params = list(grid.keys())
    grid_size = [grid[p][-1] for p in explored_params]
    chi2_grid = jnp.full(grid_size, jnp.nan)
    x_grid, y_grid = [jnp.linspace(*grid[p]) for p in explored_params]

    partial_bestfit = bestfit.copy()
    for p in explored_params:
        partial_bestfit.pop(p)

    x = flatten_vector(partial_bestfit)
    wres_, J = gauss_newton_prep(wres, partial_bestfit)

    total_points = grid_size[0] * grid_size[1]
    with tqdm(total=total_points, desc="Exploring contour") as pbar:
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                point = {explored_params[0]: x_grid[i], explored_params[1]: y_grid[j]}
                x, ploss = gauss_newton_partial(wres_, J, x, point)
                chi2_grid = chi2_grid.at[i, j].set(ploss["loss"][-1])
                pbar.update(1)
    return {
        "params": explored_params,
        "x": x_grid,
        "y": y_grid,
        "chi2": chi2_grid,
        "bestfit": bestfit,
        "extra": extra,
    }


def frequentist_contour_2D_sparse(
    likelihoods,
    grid={"Omega_m": [0.18, 0.48, 30], "w": [-0.6, -1.5, 30]},
    varied=[],
    fixed=None,
    chi2_threshold=6.17,  # 95% confidence for 2 parameters; adjust as needed
):
    """
    Compute 2D confidence contours using sparse exploration.

    Explores a grid starting from the best-fit point, stopping at a Δχ² threshold,
    assuming a convex contour to optimize progress estimation. Unexplored points
    are marked as NaN in the output grid.

    Args:
        likelihoods: List of likelihood functions.
        grid: Dict defining parameter ranges and grid sizes (e.g., {"param": [min, max, n]}).
        varied: Additional parameters to vary at each grid point (fixed can be provided instead).
        fixed: Dict of fixed parameter values.
        chi2_threshold: Δχ² threshold for contour boundary (default: 6.17, 95% for 2 params).

    Returns:
        Dict with params, x, y, chi2 grid, bestfit, and extra info.
    """
    likelihood = LikelihoodSum(likelihoods)

    # Initial setup (same as before)
    params = likelihood.initial_guess(Planck18)
    if fixed is not None:
        params.update(fixed)
        wres = restrict(likelihood.weighted_residuals, fixed)
        initial_guess = params.copy()
        for p in fixed:
            initial_guess.pop(p)
    else:
        wres, initial_guess = restrict_to(
            likelihood.weighted_residuals,
            params,
            varied=list(grid.keys()) + varied,
            flat=False,
        )

    # Find global minimum
    wres_, J = gauss_newton_prep(wres, initial_guess)
    x0 = flatten_vector(initial_guess)
    xbest, extra = gauss_newton_partial(wres_, J, x0, {})
    bestfit = unflatten_vector(initial_guess, xbest)
    chi2_min = extra["loss"][-1]

    explored_params = list(grid.keys())

    # Handle the specific case of degenerate contours by fixing one of
    # the two explored parameters
    if jnp.isnan(chi2_min):
        partial_guess = initial_guess.copy()
        first_param = explored_params[0]
        point = {first_param: partial_guess.pop(first_param)}
        wres_, J = gauss_newton_prep(wres, partial_guess)
        x0 = flatten_vector(partial_guess)
        xbest, extra = gauss_newton_partial(wres_, J, x0, point)
        bestfit = dict(unflatten_vector(partial_guess, xbest), **point)
        chi2_min = extra["loss"][-1]

    # Grid setup
    grid_size = [grid[p][-1] for p in explored_params]
    chi2_grid = jnp.full(grid_size, jnp.inf)  # Initialize with infinity
    x_grid, y_grid = [jnp.linspace(*grid[p]) for p in explored_params]

    # Find grid point closest to best-fit
    x_idx = jnp.argmin(jnp.abs(x_grid - bestfit[explored_params[0]])).item()
    y_idx = jnp.argmin(jnp.abs(y_grid - bestfit[explored_params[1]])).item()

    # Prepare for optimization
    partial_bestfit = bestfit.copy()
    for p in explored_params:
        partial_bestfit.pop(p)
    x = flatten_vector(partial_bestfit)
    wres_, J = gauss_newton_prep(wres, partial_bestfit)

    # Iterative contour exploration using a queue
    visited = set()
    queue = deque([(x_idx, y_idx)])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, right, down, left

    # Total grid points as an upper bound
    total_points = grid_size[0] * grid_size[1]

    # Progress bar with estimated total
    with tqdm(
        total=total_points, desc="Exploring contour (upper bound estimate)"
    ) as pbar:
        while queue:
            i, j = queue.popleft()
            if (
                (i, j) in visited
                or i < 0
                or i >= grid_size[0]
                or j < 0
                or j >= grid_size[1]
            ):
                continue

            visited.add((i, j))

            # Calculate chi2 at this point
            point = {explored_params[0]: x_grid[i], explored_params[1]: y_grid[j]}
            x, ploss = gauss_newton_partial(wres_, J, x, point)
            chi2_value = ploss["loss"][-1]
            chi2_grid = chi2_grid.at[i, j].set(chi2_value)

            pbar.update(1)

            # If chi2 is below threshold, explore neighbors
            if (chi2_value - chi2_min) <= chi2_threshold:
                for di, dj in directions:
                    next_i, next_j = i + di, j + dj
                    if (next_i, next_j) not in visited:
                        queue.append((next_i, next_j))

    # Convert unexplored points back to nan
    chi2_grid = jnp.where(chi2_grid == jnp.inf, jnp.nan, chi2_grid)

    return {
        "params": explored_params,
        "x": x_grid,
        "y": y_grid,
        "chi2": chi2_grid,
        "bestfit": bestfit,
        "extra": extra,
    }


latex_translation = {
    "Tcmb": r"$T_{cmb}$",
    "Omega_m": r"$\Omega_m$",
    "H0": r"$H_0$",
    "Omega_b_h2": r"$\Omega_b h^2$",
    "Omega_k": r"$\Omega_k$",
    "w": r"$w_0$",
    "wa": r"$w_a$",
    "m_nu": r"$\sum m_\nu$",
    "Neff": r"$N_{eff}$",
}

color_theme = ["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc"]


def plot_contours(
    grid,
    label=None,
    ax=None,
    bestfit=False,
    base_color=color_theme[0],
    filled=False,
    levels=[68.3, 95.5],
    **keys,
):
    """Plot 2D confidence contours from a chi-square grid.

    Generates contour plots (optionally filled) for a 2D parameter space, using
    Δχ² values derived from specified confidence levels. Shades are applied
    within a single hue, with lighter shades for lower confidence levels.
    Supports labeling for legends and plotting the best-fit point.

    Parameters
    ----------
    grid : dict
        Dictionary containing contour data, typically from `frequentist_contour_2D_sparse`.
        Expected keys:
        - 'params': List of two parameter names (e.g., ['Omega_m', 'w']).
        - 'x', 'y': 1D arrays of grid coordinates for the two parameters.
        - 'chi2': 2D array of χ² values (transposed in plotting).
        - 'bestfit': Dict of best-fit parameter values (used if `bestfit=True`).
        - 'extra': Dict with 'loss' key containing optimization results (last value used as χ²_min).
    label : str, optional
        Label for the contour set, used in the legend if provided.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses the current axes (`plt.gca()`).
    bestfit : bool, default=False
        If True, plots a black '+' at the best-fit point from `grid['bestfit']`.
    base_color : str, default is a light red hue.
        Base color hue for contours. Shades are derived by varying alpha.
    filled : bool, default=False
        If True, plots filled contours using `contourf` in addition to contour lines.
    levels : list of float, default=[68.3, 95.5]
        Confidence levels in percent (e.g., 68.3 for 1σ, 95.5 for 2σ). Converted to
        Δχ² thresholds for 2 degrees of freedom using `conflevel_to_delta_chi2`.
    **keys : dict
        Additional keyword arguments passed to `contour` and `contourf` (e.g., `linewidths`, `linestyles`).

    Notes
    -----
    - Δχ² is computed as `grid['chi2'].T - grid['extra']['loss'][-1]`,
      which is the loss value corresponding to the global minimum
      χ². This might be slightly smaller than `grid['chi2'].min()`.
    - Parameter names in axes labels are translated to LaTeX if present in `latex_translation`.
    - For filled contours, an invisible proxy patch is added for legend compatibility.
    """
    from matplotlib.colors import to_rgba

    x, y = grid["params"]
    if ax is None:
        ax = plt.gca()
    shades = jnp.linspace(1, 0.5, len(levels))
    colors = [to_rgba(base_color, alpha=alpha.item()) for alpha in shades]

    _levels = [conflevel_to_delta_chi2(l) for l in jnp.array(levels)]
    if filled:
        contours = ax.contourf(
            grid["x"],
            grid["y"],
            grid["chi2"].T - grid["extra"]["loss"][-1],  # grid["chi2"].min(),
            levels=[0] + _levels,
            label=label,
            colors=colors,
            **keys,
        )
        ax.add_patch(plt.Rectangle((jnp.nan, jnp.nan), 1, 1, fc=colors[0], label=label))
    contours = ax.contour(
        grid["x"],
        grid["y"],
        grid["chi2"].T - grid["extra"]["loss"][-1],  # grid["chi2"].min(),
        levels=_levels,
        label=label,
        colors=colors,
        **keys,
    )

    if bestfit:
        ax.plot(grid["bestfit"][x], grid["bestfit"][y], "k+")
    ax.set_xlabel(latex_translation[x] if x in latex_translation else x)
    ax.set_ylabel(latex_translation[y] if y in latex_translation else y)
