"""2D contour plots backend."""

from collections import deque
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

from .fitter import (
    restrict_to,
    restrict,
    flatten_vector,
    gauss_newton_partial,
    gauss_newton_prep,
    unflatten_vector,
    LikelihoodSum,
)
from .tools import conflevel_to_delta_chi2
from .parameters import Planck18


def frequentist_contour_2d(
    likelihoods,
    grid=None,
    varied=None,
    fixed=None,
):
    """Fully explore a 2D parameter space to build confidence contours.

    Note:
        This can be unnecessarily slow for well-behaved connected contours.
        Consider using `frequentist_contour_2d_sparse` for a more lazy exploration.

    Args:
        likelihoods: A list of likelihood functions.
        grid (dict, optional): A dictionary defining the parameter ranges and grid
            sizes. Defaults to `{"Omega_bc": [0.18, 0.48, 30], "w": [-0.6, -1.5, 30]}`.
            The format is `{"param_name": [min, max, num_points]}`.
        varied (list, optional): A list of additional parameter names to vary at
            each grid point. Defaults to None.
        fixed (dict, optional): A dictionary of parameter names and their fixed
            values. Defaults to None.

    Returns:
        A dictionary containing:
            - "params": A list of the two explored parameter names.
            - "x": A jax.numpy.ndarray with the grid values for the first parameter.
            - "y": A jax.numpy.ndarray with the grid values for the second parameter.
            - "chi2": A jax.numpy.ndarray with the chi-squared values for each grid point.
            - "bestfit": A dictionary with the best-fit parameter values.
            - "extra": A dictionary with extra information from the minimizer.
    """
    if grid is None:
        grid = {"Omega_bc": [0.18, 0.48, 30], "w": [-0.6, -1.5, 30]}
    if varied is None:
        varied = []
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
    wres_, jac = gauss_newton_prep(wres, initial_guess)
    x0 = flatten_vector(initial_guess)
    xbest, extra = gauss_newton_partial(wres_, jac, x0, {})
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
    wres_, jac = gauss_newton_prep(wres, partial_bestfit)

    total_points = grid_size[0] * grid_size[1]
    with tqdm(total=total_points, desc=f"Exploring contour {explored_params}") as pbar:
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                point = {explored_params[0]: x_grid[i], explored_params[1]: y_grid[j]}
                x, ploss = gauss_newton_partial(wres_, jac, x, point)
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


def frequentist_contour_2d_sparse(
    likelihoods,
    grid=None,
    varied=None,
    fixed=None,
    confidence_threshold=95,  # 95% confidence for 2 parameters; adjust as needed
):
    """Compute 2D confidence contours using sparse exploration.

    Explores a grid starting from the best-fit point and expanding outwards,
    stopping when a given Δχ² threshold is reached. This method is faster than
    a full grid scan but assumes that the desired contour is connected.
    Unexplored points are marked as NaN in the output grid.

    Note:
        This assumes that the contour is connected. Use `frequentist_contour_2d`
        if you suspect disconnected regions.

    Args:
        likelihoods: A list of likelihood functions.
        grid (dict, optional): A dictionary defining the parameter ranges and grid
            sizes. Defaults to `{"Omega_bc": [0.18, 0.48, 30], "w": [-0.6, -1.5, 30]}`.
            The format is `{"param_name": [min, max, num_points]}`.
        varied (list, optional): A list of additional parameter names to vary at
            each grid point. Defaults to None.
        fixed (dict, optional): A dictionary of parameter names and their fixed
            values. Defaults to None.
        confidence_threshold (float, optional): The desired confidence level in
            percent. A Δχ² threshold is computed from this value assuming 2
            degrees of freedom. Defaults to 95.

    Returns:
        A dictionary containing:
            - "params": A list of the two explored parameter names.
            - "x": A jax.numpy.ndarray with the grid values for the first parameter.
            - "y": A jax.numpy.ndarray with the grid values for the second parameter.
            - "chi2": A jax.numpy.ndarray with the chi-squared values for each grid point.
            - "bestfit": A dictionary with the best-fit parameter values.
            - "extra": A dictionary with extra information from the minimizer.
    """
    if grid is None:
        grid = {"Omega_bc": [0.18, 0.48, 30], "w": [-0.6, -1.5, 30]}
    if varied is None:
        varied = []

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
    wres_, jac = gauss_newton_prep(wres, initial_guess)
    x0 = flatten_vector(initial_guess)
    xbest, extra = gauss_newton_partial(wres_, jac, x0, {})
    bestfit = unflatten_vector(initial_guess, xbest)
    chi2_min = extra["loss"][-1]

    explored_params = list(grid.keys())

    # Handle the specific case of degenerate contours by fixing one of
    # the two explored parameters
    if jnp.isnan(chi2_min):
        partial_guess = initial_guess.copy()
        first_param = explored_params[0]
        point = {first_param: partial_guess.pop(first_param)}
        wres_, jac = gauss_newton_prep(wres, partial_guess)
        x0 = flatten_vector(partial_guess)
        xbest, extra = gauss_newton_partial(wres_, jac, x0, point)
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
    wres_, jac = gauss_newton_prep(wres, partial_bestfit)

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
        total=total_points,
        desc=f"Exploring contour {explored_params} (upper bound estimate)",
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
            x, ploss = gauss_newton_partial(wres_, jac, x, point)
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


def frequentist_1d_profile(
    likelihoods,
    grid=None,
    fixed=None,
    confidence_threshold=99.74,  # 95% confidence for 2 parameters; adjust as needed
):
    """Compute a 1D likelihood profile.

    Explores a 1D grid for a single parameter, starting from the best-fit
    point and stopping at a Δχ² threshold. Unexplored points are marked as NaN
    in the output grid.

    Note:
        This assumes that the region above `confidence_threshold` is connected.
        Use a `confidence_threshold` of 100 for a full scan if in doubt.

    Args:
        likelihoods: A list of likelihood functions.
        grid (dict, optional): A dictionary defining the parameter range and grid
            size for the single parameter to be profiled.
            Defaults to `{"Omega_bc": []}`. The format is
            `{"param_name": [min, max, num_points]}`.
        fixed (dict, optional): A dictionary of parameter names and their fixed
            values. Defaults to None.
        confidence_threshold (float, optional): The desired confidence level in
            percent. A Δχ² threshold is computed from this value assuming 1
            degree of freedom. Defaults to 99.74 (3 sigma).

    Returns:
        A dictionary containing:
            - "params": A list containing the name of the explored parameter.
            - "x": A jax.numpy.ndarray with the grid values for the parameter.
            - "chi2": A jax.numpy.ndarray with the chi-squared values for each grid point.
            - "bestfit": A dictionary with the best-fit parameter values.
            - "extra": A dictionary with extra information from the minimizer.
    """
    if grid is None:
        grid = {"Omega_bc": []}
    chi2_threshold = conflevel_to_delta_chi2(confidence_threshold, 1)

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
            varied=list(grid.keys()),
            flat=False,
        )

    # Find global minimum
    wres_, jac = gauss_newton_prep(wres, initial_guess)
    x0 = flatten_vector(initial_guess)
    xbest, extra = gauss_newton_partial(wres_, jac, x0, {})
    bestfit = unflatten_vector(initial_guess, xbest)
    chi2_min = extra["loss"][-1]

    explored_param = list(grid.keys())[0]

    # Grid setup
    grid_size = grid[explored_param][-1]
    chi2_grid = jnp.full(grid_size, jnp.inf)  # Initialize with infinity
    x_grid = jnp.linspace(*grid[explored_param])

    # Find grid point closest to best-fit
    x_idx = jnp.argmin(jnp.abs(x_grid - bestfit[explored_param])).item()

    # Prepare for optimization
    partial_bestfit = bestfit.copy()
    partial_bestfit.pop(explored_param)
    x = flatten_vector(partial_bestfit)
    wres_, jac = gauss_newton_prep(wres, partial_bestfit)

    # Iterative contour exploration using a queue
    visited = set()
    queue = deque([x_idx])
    directions = [1, -1]  # right, left

    # Progress bar with estimated total
    with tqdm(total=grid_size, desc="Exploring contour (upper bound estimate)") as pbar:
        while queue:
            i = queue.popleft()
            if i in visited or i < 0 or i >= grid_size:
                continue

            visited.add(i)

            # Calculate chi2 at this point
            point = {explored_param: x_grid[i]}
            x, ploss = gauss_newton_partial(wres_, jac, x, point)
            chi2_value = ploss["loss"][-1]
            chi2_grid = chi2_grid.at[i].set(chi2_value)

            pbar.update(1)

            # If chi2 is below threshold, explore neighbors
            if (chi2_value - chi2_min) <= chi2_threshold:
                for di in directions:
                    next_i = i + di
                    if next_i not in visited:
                        queue.append(next_i)
            # Trim down the estimation of the fraction of the plane to
            # visit when we encounter a contour boundary based on the
            # assumption that the contour is convex. This improve the
            # report of time remaining but does not affect the actual
            # exploration, which remains complete even if the contour
            # is not convex (as long as it is connected).
            else:
                if (chi2_grid[i - 1] - chi2_min) <= chi2_threshold:
                    pbar.total = pbar.total - (i - 1)
                if (chi2_grid[i + 1] - chi2_min) <= chi2_threshold:
                    pbar.total = pbar.total - (grid_size - i + 1)
                pbar.refresh()
    # Convert unexplored points back to nan
    chi2_grid = jnp.where(chi2_grid == jnp.inf, jnp.nan, chi2_grid)
    return {
        "params": [explored_param],
        "x": x_grid,
        "chi2": chi2_grid,
        "bestfit": bestfit,
        "extra": extra,
    }
