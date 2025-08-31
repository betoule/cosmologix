"""Plotting functions."""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba
import numpy as np
import jax.numpy as jnp
import jax

from cosmologix.tools import conflevel_to_delta_chi2, load

color_theme = ["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc"]

latex_translation = {
    "Tcmb": r"$T_{cmb}$",
    "Omega_m": r"$\\Omega_m$",
    "Omega_bc": r"$\\Omega_{bc}$",
    "H0": r"$H_0$",
    "Omega_b_h2": r"$\\Omega_b h^2$",
    "Omega_k": r"$\\Omega_k$",
    "w": r"$w_0$",
    "wa": r"$w_a$",
    "m_nu": r"$\\sum m_\\nu$",
    "Neff": r"$N_{eff}$",
    "M": r"$M_B$",
}


def detf_fom(result):
    """Computes the Dark Energy Task Force (DETF) Figure of Merit (FoM).

    The FoM is calculated as the inverse of the square root of the
    determinant of the w-wa covariance matrix.

    Args:
        result (dict): A dictionary containing the Fisher matrix results,
            including 'bestfit' and 'inverse_FIM'.

    Returns:
        float: The DETF Figure of Merit.
    """
    bestfit = result["bestfit"]
    ifim = result["inverse_FIM"]  # covariance matrix

    # Parameter names (assuming they match the order in FIM)
    param_names = list(bestfit.keys())

    # Retrieve indexes corresponding to param1 and 2
    index = [param_names.index("w"), param_names.index("wa")]

    return 1.0 / np.sqrt(np.linalg.det(ifim[np.ix_(index, index)]))


def pretty_print(result):
    """Pretty-prints best-fit parameters with uncertainties.

    Args:
        result (dict): A dictionary containing the fit results, including
            'bestfit', 'inverse_FIM', 'loss', and 'residuals'.
    """
    bestfit = result["bestfit"]
    ifim = result["inverse_FIM"]  # covariance matrix

    # Uncertainties are sqrt of diagonal elements of covariance matrix
    uncertainties = jnp.sqrt(jnp.diag(ifim))

    # Parameter names (assuming they match the order in FIM)
    param_names = list(bestfit.keys())

    # Print each parameter with its uncertainty
    for i, (param, value) in enumerate(bestfit.items()):
        uncertainty = uncertainties[i]
        if uncertainty == 0:  # Avoid log(0)
            precision = 3  # Default if no uncertainty
        else:
            # Number of decimal places to align with first significant digit of uncertainty
            precision = max(0, -int(jnp.floor(jnp.log10(abs(uncertainty)))) + 1)
        fmt = f"{{:.{{precision}}f}}"
        print(f"{param} = {fmt.format(value)} ± {fmt.format(uncertainty)}")
    chi2 = result["loss"][-1]
    residuals = result["residuals"]
    ndof = len(residuals) - len(param_names)
    pvalue = 1 - jax.scipy.stats.chi2.cdf(chi2, ndof)
    print(f"χ²={chi2:.2f} (d.o.f. = {ndof}), χ²/d.o.f = {chi2/ndof:.3f}")
    # If the fit involves w and wa print the FOM
    print(f"p-value: {pvalue*100:.2f}%")
    if "w" in param_names and "wa" in param_names:
        print(f"FOM={detf_fom(result):.1f}")


def plot_confidence_ellipse(
    mean, cov, ax=None, n_sigmas=None, color=color_theme[0], **kwargs
):
    """Plots a confidence ellipse for two parameters.

    Args:
        mean (array-like): Mean values of the two parameters, shape (2,).
        cov (array-like): 2x2 covariance matrix of the two parameters.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, uses
            the current axes. Defaults to None.
        n_sigmas (list, optional): A list of sigma values for the ellipses.
            Defaults to [1.5, 2.5].
        color (str, optional): The color of the ellipse. Defaults to the first
            color in the theme.
        **kwargs: Additional keyword arguments passed to `matplotlib.patches.Ellipse`.

    Returns:
        matplotlib.patches.Ellipse: The plotted ellipse object.
    """
    if n_sigmas is None:
        n_sigmas = [1.5, 2.5]
    if ax is None:
        ax = plt.gca()

    # Eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]  # Sort descending
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    # Width and height of the ellipse (2 * sqrt(eigenvalues) for 1σ)
    width, height = 2 * np.sqrt(eigenvalues)

    # Angle of rotation in degrees (from eigenvector)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    for n_sigma, alpha in zip(n_sigmas, np.linspace(1, 0.5, len(n_sigmas))):
        # Create the ellipse
        ellipse = Ellipse(
            xy=mean,
            width=width * n_sigma,
            height=height * n_sigma,
            angle=angle,
            edgecolor=color,
            fill=False,
            alpha=alpha,
            **kwargs,
        )
        # Add to plot
        ax.add_patch(ellipse)

    return ellipse


def plot_1d(
    result,
    param,
    ax=None,
    color=color_theme[0],
):
    """Plots a 1D Gaussian distribution from a Fisher matrix result.

    Args:
        result (dict): A dictionary containing the Fisher matrix results.
        param (str): The name of the parameter to plot.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, uses
            the current axes. Defaults to None.
        color (str, optional): The color of the plot. Defaults to the first
            color in the theme.
    """
    if ax is None:
        ax = plt.gca()
        ax.set_xlabel(latex_translation[param])
    bestfit = result["bestfit"]
    ifim = result["inverse_FIM"]  # covariance matrix

    # Parameter names (assuming they match the order in FIM)
    param_names = list(bestfit.keys())

    # Retrieve indexes corresponding to param
    index = param_names.index(param)

    # select the relevant part the results
    sigma = np.sqrt(ifim[index, index])
    mean = bestfit[param]
    x = np.linspace(mean - 3 * sigma, mean + 3 * sigma)
    ax.plot(x, np.exp(-0.5 * (x - mean) ** 2 / sigma**2), color=color)


def plot_2d(
    result,
    param1,
    param2,
    ax=None,
    n_sigmas=None,
    marker="s",
    color=color_theme[0],
    **kwargs,
):
    """Plots a 2D confidence ellipse from a Fisher matrix result.

    Args:
        result (dict): A dictionary containing the Fisher matrix results.
        param1 (str): The name of the first parameter.
        param2 (str): The name of the second parameter.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, uses
            the current axes. Defaults to None.
        n_sigmas (list, optional): A list of sigma values for the ellipses.
            Defaults to [1.5, 2.5].
        marker (str, optional): The marker for the best-fit point.
            Defaults to 's'.
        color (str, optional): The color of the plot. Defaults to the first
            color in the theme.
        **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.plot`.
    """
    if n_sigmas is None:
        n_sigmas = [1.5, 2.5]
    if ax is None:
        ax = plt.gca()
    bestfit = result["bestfit"]
    ifim = result["inverse_FIM"]  # covariance matrix

    # Parameter names (assuming they match the order in FIM)
    param_names = list(bestfit.keys())

    # Retrieve indexes corresponding to param1 and 2
    index = [param_names.index(param1), param_names.index(param2)]

    # select the block of the covariance matrix
    cov = ifim[np.ix_(index, index)]

    #
    mean = (bestfit[param1], bestfit[param2])

    ax.plot(*mean, marker=marker, ls="None", color=color, **kwargs)
    plot_confidence_ellipse(mean, cov, ax=ax, n_sigmas=n_sigmas, color=color, **kwargs)


def plot_profile(
    grid,
    label=None,
    filled=False,
    ax=None,
    color=color_theme[0],
):
    """Plots a 1D profile likelihood from a chi-squared vector.

    Args:
        grid (dict or str): A dictionary or path to a pickle file containing
            the grid data from `frequentist_1d_profile`.
        label (str, optional): Label for the plot. Defaults to None.
        filled (bool, optional): Whether to fill the area under the curve.
            Defaults to False.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, uses
            the current axes. Defaults to None.
        color (str, optional): The color of the plot. Defaults to the first
            color in the theme.
    """
    grid = load(grid)

    param = grid["params"][0]
    chi2_min = grid["extra"]["loss"][-1]
    if ax is None:
        ax = plt.gca()
        ax.set_xlabel(latex_translation[param])
    if filled:
        ax.fill_between(
            grid["x"],
            jnp.exp(-0.5 * (grid["chi2"] - chi2_min)),
            y2=0,
            color=color,
            alpha=0.5,
        )
    ax.plot(
        grid["x"], jnp.exp(-0.5 * (grid["chi2"] - chi2_min)), color=color, label=label
    )


def plot_contours(
    grid,
    label=None,
    ax=None,
    bestfit=False,
    color=color_theme[0],
    filled=False,
    transpose=False,
    levels=None,
    **keys,
):
    """Plots 2D confidence contours from a chi-square grid.

    Generates contour plots (optionally filled) for a 2D parameter space,
    using Δχ² values derived from specified confidence levels.

    Args:
        grid (dict or str): A dictionary or path to a pickle file containing
            the grid data from `frequentist_contour_2d_sparse`.
        label (str, optional): Label for the contour set. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, uses
            the current axes. Defaults to None.
        bestfit (bool, optional): Whether to plot the best-fit point.
            Defaults to False.
        color (str, optional): The base color for the contours. Defaults to
            the first color in the theme.
        filled (bool, optional): Whether to fill the contours. Defaults to False.
        transpose (bool, optional): Whether to transpose the x and y axes.
            Defaults to False.
        levels (list, optional): A list of confidence levels in percent.
            Defaults to [68.3, 95.5].
        **keys: Additional keyword arguments passed to `matplotlib.pyplot.contour`
            and `matplotlib.pyplot.contourf`.
    """
    if levels is None:
        levels = [68.3, 95.5]

    grid = load(grid)

    x, y = grid["params"]
    if transpose:
        x, y = y, x
        xl = "y"
        yl = "x"
        values = grid["chi2"]
    else:
        xl = "x"
        yl = "y"
        values = grid["chi2"].T
    if ax is None:
        ax = plt.gca()
        ax.set_xlabel(latex_translation[x] if x in latex_translation else x)
        ax.set_ylabel(latex_translation[y] if y in latex_translation else y)

    shades = jnp.linspace(1, 0.5, len(levels))
    colors = [to_rgba(color, alpha=alpha.item()) for alpha in shades]

    if ("label" in grid) and label is None:
        label = grid["label"]
    _levels = [conflevel_to_delta_chi2(l) for l in jnp.array(levels)]
    if filled:
        ax.contourf(
            grid[xl],
            grid[yl],
            values - grid["extra"]["loss"][-1],  # grid["chi2"].min(),
            levels=[0] + _levels,
            colors=colors,
            **keys,
        )
        ax.add_patch(plt.Rectangle((jnp.nan, jnp.nan), 1, 1, fc=colors[0], label=label))
    else:
        ax.add_line(plt.Line2D((jnp.nan,), (jnp.nan,), color=colors[0], label=label))
    ax.contour(
        grid[xl],
        grid[yl],
        values - grid["extra"]["loss"][-1],  # grid["chi2"].min(),
        levels=_levels,
        colors=colors,
        **keys,
    )

    if bestfit:
        ax.plot(grid["bestfit"][x], grid["bestfit"][y], "k+")


def corner_plot(param_names, axes=None, figsize=(10, 10)):
    """Creates a corner plot grid for visualizing parameter distributions.

    This function sets up a triangular grid of subplots for a corner plot.

    Args:
        param_names (list): A list of parameter names to define the grid size
            and labels.
        axes (numpy.ndarray, optional): A pre-existing array of axes to
            populate. If None, a new figure and axes are created.
            Defaults to None.
        figsize (tuple, optional): The figure size. Defaults to (10, 10).

    Returns:
        numpy.ndarray: An array of matplotlib axes objects.
    """
    if axes is None:
        fig = plt.figure(figsize=figsize)
        axes = fig.subplots(
            len(param_names), len(param_names), sharex="col", squeeze=False
        )
    for i, param in enumerate(param_names):
        for j, param2 in enumerate(param_names):
            if i == j:
                axes[i, i].spines["left"].set_visible(False)
                axes[i, i].spines["right"].set_visible(False)
                axes[i, i].spines["top"].set_visible(False)
                axes[j, i].set_yticks([])
            elif j > i:
                pass
            else:
                axes[j, i].set_visible(False)
            if j == len(param_names) - 1:
                axes[j, i].set_xlabel(latex_translation[param])
            if i == 0:
                if j > 0:
                    axes[j, i].set_ylabel(latex_translation[param2])
            else:
                axes[j, i].set_yticks([])
    plt.tight_layout()
    return axes


def corner_plot_fisher(results, param_names=None, axes=None, **keys):
    """Plots 1D and 2D Fisher matrix distributions on a corner plot grid.

    This function overlays 1D Gaussian distributions on the diagonal and 2D
    confidence ellipses in the lower triangle of a corner plot, based on
    Fisher matrix results.

    Args:
        results (dict): A dictionary containing Fisher matrix results.
        param_names (list, optional): A list of parameter names to plot. If
            None, they are extracted from `results['bestfit'].keys()`.
            Defaults to None.
        axes (numpy.ndarray, optional): A pre-existing array of axes. If None,
            a new one is created. Defaults to None.
        **keys: Additional keyword arguments passed to `plot_1d` and `plot_2d`.

    Returns:
        tuple: A tuple containing the axes and the list of parameter names.
    """
    if param_names is None:
        param_names = list(results["bestfit"].keys())
    if axes is None:
        axes = corner_plot(param_names)

    for i, param in enumerate(param_names):
        for j, param2 in enumerate(param_names):
            if i == j:
                plot_1d(results, param, ax=axes[i, i], **keys)
            elif j > i:
                plot_2d(results, param, param2, ax=axes[j, i], **keys)
    return axes, param_names


def corner_plot_contours(grids=None, axes=None, param_names=None, **keys):
    """Plots 2D contour grids on a corner plot for multiple parameter pairs.

    This function adds 2D contour plots to the lower triangle of a corner plot,
    using precomputed grid data from likelihood scans.

    Args:
        grids (list, optional): A list of dictionaries, each containing grid
            data for contour plotting. Defaults to None.
        axes (numpy.ndarray, optional): A pre-existing array of axes. If None,
            a new one is created. Defaults to None.
        param_names (list, optional): A list of all parameter names. If None,
            it is extracted from the grids. Defaults to None.
        **keys: Additional keyword arguments passed to `corner_plot` and
            `plot_contours`.

    Returns:
        tuple: A tuple containing the axes and the list of parameter names.
    """
    if grids is None:
        grids = []
    if param_names is None:
        param_names = []
        for grid in grids:
            param_names.extend(grid["params"])
        param_names = list(set(param_names))
    if axes is None:
        axes = corner_plot(param_names)
    for grid in grids:
        if len(grid["params"]) == 2:
            param, param2 = grid["params"]
            i = param_names.index(param)
            j = param_names.index(param2)
            if i < j:
                plot_contours(grid, ax=axes[j, i], **keys)
            else:
                plot_contours(grid, ax=axes[i, j], transpose=True, **keys)
        else:
            param = grid["params"][0]
            i = param_names.index(param)
            plot_profile(grid, ax=axes[i, i], **keys)
    return axes, param_names