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

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            print(i, j)
            point = {explored_params[0]: x_grid[i], explored_params[1]: y_grid[j]}
            x, ploss = gauss_newton_partial(wres_, J, x, point)
            chi2_grid = chi2_grid.at[i, j].set(ploss["loss"][-1])
    return {
        "params": explored_params,
        "x": x_grid,
        "y": y_grid,
        "chi2": chi2_grid,
        "bestfit": bestfit,
        "extra": extra,
    }


from collections import deque


def frequentist_contour_2D_sparse(
    likelihoods,
    grid={"Omega_m": [0.18, 0.48, 30], "w": [-0.6, -1.5, 30]},
    varied=[],
    fixed=None,
    chi2_threshold=6.17,  # 95% confidence for 2 parameters; adjust as needed
):
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

    # Find global minimum (same as before)
    wres_, J = gauss_newton_prep(wres, initial_guess)
    x0 = flatten_vector(initial_guess)
    xbest, extra = gauss_newton_partial(wres_, J, x0, {})
    bestfit = unflatten_vector(initial_guess, xbest)
    chi2_min = extra["loss"][-1]

    # Grid setup
    explored_params = list(grid.keys())
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


def plot_contours(
    grid,
    label=None,
    ax=None,
    bestfit=False,
    base_color="blue",
    filled=False,
    levels=[68.3, 95.5],
    **keys,
):
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
    ax.set_xlabel(x)
    ax.set_ylabel(y)


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)
    from cosmologix import likelihoods

    priors = [likelihoods.Planck2018Prior(), likelihoods.DES5yr()]
    fixed = {"Omega_k": 0.0, "m_nu": 0.06, "Neff": 3.046, "Tcmb": 2.7255, "wa": 0.0}
    n = 60
    grid = frequentist_contour_2D_sparse(
        priors, grid={"Omega_m": [0.18, 0.48, n], "w": [-0.6, -1.5, n]}, fixed=fixed
    )
    grid_sn = frequentist_contour_2D_sparse(
        [likelihoods.DES5yr()],
        grid={"Omega_m": [0.18, 0.48, n], "w": [-0.6, -1.5, n]},
        fixed=dict(fixed, H0=Planck18["H0"], Omega_b_h2=Planck18["Omega_b_h2"]),
    )

    grid_cmb = frequentist_contour_2D_sparse(
        [likelihoods.Planck2018Prior()],
        grid={"Omega_m": [0.18, 0.48, n], "w": [-0.6, -1.5, n]},
        fixed=fixed,
    )
    # varied=['M', 'Omega_b_h2', 'H0'])
    # grid = frequentist_contour_2D([DES5yr()], varied=['M'])
    plot_contours(grid)
    import matplotlib.pyplot as plt

    plot_contours(grid_cmb, base_color="green", filled=True, label="Planck")
    plot_contours(grid_sn, base_color="blue", filled=True, label="DES5")
    plot_contours(
        grid, base_color="black", filled=False, label="Planck+DES5", bestfit=True
    )
    plt.legend(loc="lower right", frameon=False)
    plt.show()
