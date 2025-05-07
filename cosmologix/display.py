"""
Plotting functions
"""

from pathlib import Path
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
    "Omega_m": r"$\Omega_m$",
    "Omega_bc": r"$\Omega_{bc}$",
    "H0": r"$H_0$",
    "Omega_b_h2": r"$\Omega_b h^2$",
    "Omega_k": r"$\Omega_k$",
    "w": r"$w_0$",
    "wa": r"$w_a$",
    "m_nu": r"$\sum m_\nu$",
    "Neff": r"$N_{eff}$",
    "M": r"$M_B$",
}


def detf_fom(result):
    """Compute the dark energy task force figure of merit as the
    inverse of the square root of the determinant of the w wa
    covariance.
    """
    bestfit = result["bestfit"]
    ifim = result["inverse_FIM"]  # covariance matrix

    # Parameter names (assuming they match the order in FIM)
    param_names = list(bestfit.keys())

    # Retrieve indexes corresponding to param1 and 2
    index = [param_names.index("w"), param_names.index("wa")]

    return 1.0 / np.sqrt(np.linalg.det(ifim[np.ix_(index, index)]))


def pretty_print(result):
    """Pretty-print best-fit parameters with uncertainties from the Fisher Information Matrix."""
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
        fmt = f"{{:.{precision}f}}"
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
    """Plot a confidence ellipse for two parameters given their mean and covariance.

    Parameters
    ----------
    mean : array-like
        Mean values of the two parameters, shape (2,) (e.g., [x_mean, y_mean]).
    cov : array-like
        2x2 covariance matrix of the two parameters.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; if None, creates a new figure.
    n_sigma : float, optional
        Number of standard deviations for the ellipse (e.g., 1 for 1σ, 2 for 2σ).
    **kwargs : dict
        Additional keyword arguments passed to Ellipse (e.g., facecolor, edgecolor).

    Returns
    -------
    matplotlib.patches.Ellipse
        The plotted ellipse object.
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
    """1D gaussian"""
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
    """2D ellipse"""
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
    ax=None,
    color=color_theme[0],
):
    """Plot a 1D profile likelihood from a chi2 vector.

    Parameters
    ----------
    grid:  dict or str or path
        Dictionary or path to a pickle file containing a dictionary.
        The dictionary contains contour data, typically from `frequentist_1d_profile`.
        Expected keys:
        - 'params': List of 1 parameter name (e.g., ['Omega_bc']).
        - 'x': 1D array of grid coordinates for the explored parameter.
        - 'chi2': 1D array of χ² values.
        - 'bestfit': Dict of best-fit parameter values (used if `bestfit=True`).
        - 'extra': Dict with 'loss' key containing optimization results (last value used as χ²_min).
    label : str, optional
        Label for the contour set, used in the legend if provided.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses the current axes (`plt.gca()`).
    color : str, default is a light red hue.
    """
    if isinstance(grid, (str, Path)):
        grid = load(grid)

    param = grid["params"][0]
    chi2_min = grid["extra"]["loss"][-1]
    if ax is None:
        ax = plt.gca()
        ax.set_xlabel(latex_translation[param])
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
    """Plot 2D confidence contours from a chi-square grid.

    Generates contour plots (optionally filled) for a 2D parameter space, using
    Δχ² values derived from specified confidence levels. Shades are applied
    within a single hue, with lighter shades for lower confidence levels.
    Supports labeling for legends and plotting the best-fit point.

    Parameters
    ----------
    grid : dict or str or path
        Dictionary or path to a pickle file containing a dictionary.
        The dictionary contains contour data, typically from `frequentist_contour_2d_sparse`.
        Expected keys:
        - 'params': List of two parameter names (e.g., ['Omega_bc', 'w']).
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
    color : str, default is a light red hue.
        Base color hue for contours. Shades are derived by varying alpha.
    filled : bool, default=False
        If True, plots filled contours using `contourf` in addition to contour lines.
    levels : list of float, default=[68.3, 95.5]
        Confidence levels in percent (e.g., 68.3 for 1σ, 95.5 for 2σ). Converted to
        Δχ² thresholds for 2 degrees of freedom using `conflevel_to_delta_chi2`.
    transpose: bool, default=False
        Exchange x and y parameters when plotting
    **keys : dict
        Additional keyword arguments passed to `contour` and `contourf`
        (e.g., `linewidths`, `linestyles`).

    Notes
    -----
    - Δχ² is computed as `grid['chi2'].T - grid['extra']['loss'][-1]`,
      which is the loss value corresponding to the global minimum
      χ². This might be slightly smaller than `grid['chi2'].min()`.
    - Parameter names in axes labels are translated to LaTeX if present in `latex_translation`.
    - For filled contours, an invisible proxy patch is added for legend compatibility.
    """
    if levels is None:
        levels = [68.3, 95.5]

    if isinstance(grid, (str, Path)):
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


def corner_plot(param_names, axes=None, figsize=(12, 12)):
    """Create a corner plot grid for visualizing parameter distributions.

    This function sets up a triangular grid of subplots for a corner plot, where the
    diagonal contains 1D histograms and the lower triangle can hold 2D scatter or
    contour plots. The upper triangle is suppressed, and y-axis ticks are removed
    except in the first column. Spines are adjusted on the diagonal for a clean look.

    Parameters
    ----------
    param_names : list
        List of parameter names to define the grid size and labels (e.g., ['Omega_bc', 'H0']).
    axes : numpy.ndarray, optional
        Pre-existing array of axes to populate; if None, a new figure and axes are created.
    figsize : (float, float) default (12,12)
        figure dimension passed to figure or subplot creation.

    Returns
    -------
    numpy.ndarray
        Array of matplotlib axes objects, shape (n, n), where n is the length of param_names.

    Notes
    -----
    - The diagonal histograms have left, right, and top spines removed for aesthetic clarity.
    - X-axis labels are set only in the bottom row, and y-axis labels only in the first column.
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
    """Plot 1D and 2D Fisher matrix distributions on a corner plot grid.

    This function overlays 1D Gaussian distributions on the diagonal and 2D confidence
    ellipses in the lower triangle of a corner plot, based on Fisher matrix results.
    It builds on `corner_plot` for the grid layout and uses helper functions `plot_1D`
    and `plot_2d` for the actual plotting.

    Parameters
    ----------
    results : dict
        Dictionary containing Fisher matrix results, with 'bestfit' (means) and covariance data.
    param_names : list, optional
        List of parameter names to plot; if None, extracted from results['bestfit'].keys().
    axes : numpy.ndarray, optional
        Pre-existing array of axes; if None, created via `corner_plot`.
    **keys : dict
        Additional keyword arguments passed to `plot_1d`, and `plot_2d`.

    Returns
    -------
    tuple
        (axes, param_names), where axes is the array of matplotlib axes and param_names is the
        list of parameters plotted.
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
    """Plot 2D contour grids on a corner plot for multiple parameter pairs.

    This function adds 2D contour plots to the lower triangle of a corner plot, using
    precomputed grid data (from likelihood scans). It builds on `corner_plot`
    and uses `plot_contours` to render each contour.

    Parameters
    ----------
    grids : list, optional
        List of dictionaries, each containing 'params' (tuple of two parameter names),
        'bestfit', and grid data for contour plotting.
    axes : numpy.ndarray, optional
        Pre-existing array of axes; if None, created via `corner_plot`.
    param_names : list, optional
        List of all parameter names; if None, extracted from
    **keys : dict
        Additional keyword arguments passed to `corner_plot` and `plot_contours`.

    Returns
    -------
    tuple
        (axes, param_names), where axes is the array of matplotlib axes and param_names is the
        list of parameters in the grid.
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
