from cosmologix.likelihoods import LikelihoodSum
from cosmologix.fitter import (
    restrict_to,
    restrict,
    partial,
    flatten_vector,
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
from cosmologix.display import color_theme, latex_translation, plot_contours
import numpy as np


def frequentist_contour_2D(
    likelihoods,
    grid={"Omega_m": [0.18, 0.48, 30], "w": [-0.6, -1.5, 30]},
    varied=[],
    fixed=None,
):
    """Full explore a 2D parameter space to build confidence contours.

    Note: This can be unecessary slow for well behaved connected
    contours. Have a look to Use frequentist_contour_2D_sparse for a
    more lazy exploration.

    Args:
        likelihoods: List of likelihood functions.
        grid: Dict defining parameter ranges and grid sizes (e.g., {"param": [min, max, n]}).
        varied: Additional parameters to vary at each grid point (fixed can be provided instead).
        fixed: Dict of fixed parameter values.

    Returns:
        Dict with params, x, y, chi2 grid, bestfit, and extra info.

    """
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
    confidence_threshold=95,  # 95% confidence for 2 parameters; adjust as needed
):
    """
    Compute 2D confidence contours using sparse exploration.

    Explores a grid starting from the best-fit point, stopping at a Δχ² threshold,
    assuming a convex contour to optimize progress estimation. Unexplored points
    are marked as NaN in the output grid.

    Important Note:
    This assumes that the contour is connected. Use frequentist_contour_2D when in doubt.

    Args:
        likelihoods: List of likelihood functions.
        grid: Dict defining parameter ranges and grid sizes (e.g., {"param": [min, max, n]}).
        varied: Additional parameters to vary at each grid point (fixed can be provided instead).
        fixed: Dict of fixed parameter values.
        chi2_threshold: largest confidence level in percent for contour boundary. A Δχ² threshold is computed for this value assuming 2 degrees of freedom. (default: 95% corresponding to 6.17 for 2 params).

    Returns:
        Dict with params, x, y, chi2 grid, bestfit, and extra info.
    """
    chi2_threshold = conflevel_to_delta_chi2(confidence_threshold)

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

    # Exploration progress
    exploration_progress = np.ones(grid_size, dtype="bool")

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
            # Trim down the estimation of the fraction of the plane to
            # visit when we encounter a contour boundary based on the
            # assumption that the contour is convex. This improve the
            # report of time remaining but does not affect the actual
            # exploration, which remains complete even if the contour
            # is not convex (as long as it is connected).
            else:
                if (chi2_grid[i - 1, j] - chi2_min) <= chi2_threshold:
                    exploration_progress[i + 1 :, j] = False
                    if (chi2_grid[i, j - 1] - chi2_min) <= chi2_threshold:
                        exploration_progress[i + 1 :, j + 1 :] = False
                    if (chi2_grid[i, j + 1] - chi2_min) <= chi2_threshold:
                        exploration_progress[i + 1 :, : j - 1] = False
                if (chi2_grid[i + 1, j] - chi2_min) <= chi2_threshold:
                    exploration_progress[: i - 1, j] = False
                    if (chi2_grid[i, j + 1] - chi2_min) <= chi2_threshold:
                        exploration_progress[: i - 1, : j - 1] = False
                    if (chi2_grid[i, j - 1] - chi2_min) <= chi2_threshold:
                        exploration_progress[: i - 1, j + 1 :] = False
                if (chi2_grid[i, j - 1] - chi2_min) <= chi2_threshold:
                    exploration_progress[i, j + 1 :] = False
                if (chi2_grid[i, j + 1] - chi2_min) <= chi2_threshold:
                    exploration_progress[i, : j - 1] = False
                pbar.total = exploration_progress.sum()
                pbar.refresh()
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


def save_contours(grid, filename):
    """Save contour data dictionary to a pickle file."""
    import pickle

    with open(filename, "wb") as fid:
        pickle.dump(grid, fid)


def load_contours(filename):
    """Load contour data dictionary from a pickle file."""
    import pickle

    with open(filename, "rb") as fid:
        return pickle.load(fid)
