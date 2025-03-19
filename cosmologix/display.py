import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import jax.numpy as jnp
import jax

color_theme = ["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc"]

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


def detf_fom(result):
    """Compute the dark energy task force figure of merit as the
    inverse determinant of the w wa covariance.
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
    ndof = len(residuals) - len(param)
    pvalue = 1 - jax.scipy.stats.chi2.cdf(chi2, ndof)
    print(f"χ²={chi2:.2f} (d.o.f. = {ndof}), χ²/d.o.f = {chi2/ndof:.3f}")
    # If the fit involves w and wa print the FOM
    print(f"p-value: {pvalue*100:.2f}%")
    if "w" in param_names and "wa" in param_names:
        fom = detf_fom(result)
        print(f"FOM={fom:.1f}")


def plot_confidence_ellipse(
    mean, cov, ax=None, n_sigmas=[1.5, 2.5], color=color_theme[0], **kwargs
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

    alphas = np.linspace(1, 0.5, len(n_sigmas))
    for n_sigma, alpha in zip(n_sigmas, alphas):
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


def plot_2D(
    result,
    param1,
    param2,
    ax=None,
    n_sigmas=[1.5, 2.5],
    marker="s",
    color=color_theme[0],
    **kwargs,
):
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
