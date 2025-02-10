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


def plot_contours(grid, label=None, ax=None, bestfit=False, **keys):
    x, y = grid["params"]
    if ax is None:
        ax = plt.gca()
    ax.contour(
        grid["x"],
        grid["y"],
        grid["chi2"].T - grid["chi2"].min(),
        [conflevel_to_delta_chi2(l) for l in jnp.array([68.3, 95.5])],
        label=label,
        **keys,
    )
    if bestfit:
        ax.plot(grid["bestfit"][x], grid["bestfit"][y], "rs")
    ax.set_xlabel(x)
    ax.set_ylabel(y)


if __name__ == "__main__":
    import jax
    from cosmologix.likelihoods import DES5yr, Planck2018Prior

    jax.config.update("jax_enable_x64", True)
    fixed = Planck18.copy()
    for p in ["Omega_b_h2", "H0", "w", "Omega_m"]:
        fixed.pop(p)
    grid = frequentist_contour_2D([DES5yr(), Planck2018Prior()], fixed=fixed)
    # varied=['M', 'Omega_b_h2', 'H0'])
    # grid = frequentist_contour_2D([DES5yr()], varied=['M'])
    plot_contours(grid)
    import matplotlib.pyplot as plt

    plt.show()
