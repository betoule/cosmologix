#!/usr/bin/env python
"""Cosmologix high level interface

A collection of script for common tasks also made available from the
command line thanks to typer

"""

from typing import List, Optional, Tuple, Any
import typer
import click
from typer import Typer, Option, Argument
from cosmologix import parameters

# We defer other imports to improve responsiveness on the command line
# pylint: disable=import-outside-toplevel

# Main Typer app
app = Typer(
    name="cosmologix",
    help="Cosmological fitting tool",
    no_args_is_help=True,
    add_completion=True,  # Enable default typer completion
)

# Available priors
AVAILABLE_PRIORS = [
    "Planck2018",
    "PR4",
    "DESIDR1",
    "DESIDR2",
    "DES5yr",
    "Pantheonplus",
    "Union3",
    "SH0ES",
    "BBNNeffSchoneberg2024",
    "BBNSchoneberg2024",
]

PARAM_CHOICES = list(parameters.Planck18.keys()) + ["M", "rd"]


def tuple_list_to_dict(tuple_list):
    """Parse parameters such as --range Omega_bc 0 1 into a dict"""
    result_dict = {}
    for item in tuple_list:
        result_dict[item[0]] = list(item[1:])
    return result_dict


def dict_to_list(dictionnary):
    """Convert default dictionnaries to string usable in command line completion"""

    def to_str(v):
        try:
            return " ".join(str(_v) for _v in v)
        except TypeError:
            return str(v)

    def f():

        return [f"{k} {to_str(v)}" for k, v in dictionnary.items()]

    return f


# Shared option definitions
COSMOLOGY_OPTION = Option(
    "FwCDM",
    "--cosmology",
    "-c",
    help="Cosmological model",
    show_choices=True,
    autocompletion=lambda: list(parameters.DEFAULT_FREE.keys()),
)
PRIORS_OPTION = Option(
    [],
    "--priors",
    "-p",
    help="Priors to use (e.g., Planck18 DESI2024)",
    show_choices=True,
    autocompletion=lambda: AVAILABLE_PRIORS,
)
FIX_OPTION = Option(
    [],
    "--fix",
    "-F",
    help="Fix PARAM at VALUE (e.g., -F H0 70)",
    autocompletion=dict_to_list(parameters.Planck18),
    click_type=click.Tuple([str, float]),
)
FREE_OPTION = Option(
    [],
    "--free",
    help="Force release of parameter (e.g., --free Neff)",
    show_choices=True,
    autocompletion=lambda: PARAM_CHOICES,
)
RANGE_OPTION = Option(
    [],
    "--range",
    help="Override exploration range for a parameter (e.g., --range Omega_bc 0.1 0.5)",
    show_choices=True,
    autocompletion=dict_to_list(parameters.DEFAULT_RANGE),
    click_type=click.Tuple([str, float, float]),
)
MU_OPTION = Option(
    None,
    "--mu",
    help="Distance modulus data file in npy format",
)
MU_COV_OPTION = Option(
    None,
    "--mu-cov",
    help="Optional covariance matrix in npy format",
)


def validate_fix(value: str) -> Any:
    """Validate --fix parameter: string if in PARAM_CHOICES or 'M'/'rd', else float."""
    valid_params = PARAM_CHOICES
    if value in valid_params:
        return value
    try:
        return float(value)
    except ValueError:
        raise typer.BadParameter(f"Value must be a float or one of {valid_params}")


def get_prior(p):
    """Retrieve a prior by name"""
    import cosmologix.likelihoods

    return getattr(cosmologix.likelihoods, p)()


def load_mu(mu_file: str, cov_file: str = ""):
    """Load distance measurement."""
    if mu_file is None:
        return []
    import numpy as np
    from cosmologix import likelihoods

    muobs = np.load(mu_file)
    if cov_file:
        cov = np.load(cov_file)
        like = likelihoods.MuMeasurements(muobs["z"], muobs["mu"], cov)
    else:
        like = likelihoods.DiagMuMeasurements(muobs["z"], muobs["mu"], muobs["muerr"])
    return [like]


def auto_restricted_fit(priors, fixed, verbose):
    """Test if there is unconstrained parameters"""
    from . import fitter

    for _ in range(3):
        try:
            result = fitter.fit(priors, fixed=fixed, verbose=verbose)
            break
        except fitter.UnconstrainedParameterError as e:
            for param in e.params:
                print(f"Fixing unconstrained parameter {param[0]}")
                fixed[param[0]] = parameters.Planck18[param[0]]
        except fitter.DegenerateParametersError:
            print("Try again fixing H0")
            fixed["H0"] = parameters.Planck18["H0"]
    return result


@app.command()
def fit(
    prior_names: List[str] = PRIORS_OPTION,
    cosmology: str = COSMOLOGY_OPTION,
    verbose: bool = Option(
        False, "--verbose", "-v", help="Display the successive steps of the fit"
    ),
    fix: List[click.Tuple] = FIX_OPTION,
    free: List[str] = FREE_OPTION,
    mu: Optional[str] = MU_OPTION,
    mucov: Optional[str] = MU_COV_OPTION,
    auto_constrain: bool = Option(
        False,
        "--auto-constrain",
        "-A",
        help="Impose hard priors on unconstrained parameters (use with caution)",
    ),
    show: bool = Option(
        False, "--show", "-s", help="Display best-fit results as a corner plot"
    ),
    output: Optional[str] = Option(
        None,
        "--output",
        "-o",
        help="Output file for best-fit parameters (e.g., planck_desi.pkl)",
    ),
):
    """Find bestfit cosmological model."""
    from . import fitter, display, tools
    priors = [get_prior(p) for p in prior_names] + load_mu(mu, mucov)
    fixed = parameters.Planck18.copy()
    to_free = parameters.DEFAULT_FREE[cosmology].copy()
    for par in to_free + free:
        fixed.pop(par)
    for par, value in fix:
        fixed[par] = value
    if auto_constrain:
        result = auto_restricted_fit(priors, fixed, verbose)
    else:
        result = fitter.fit(priors, fixed=fixed, verbose=verbose)
    display.pretty_print(result)
    if output:
        result["label"] = "+".join(prior_names) + ("+SN" if mu is not None else "")
        tools.save(result, output)
        print(f"Best-fit parameters saved to {output}")
    if show:
        import matplotlib.pyplot as plt

        display.corner_plot_fisher(result)
        plt.tight_layout()
        plt.show()


@app.command()
def explore(
    params: List[str] = Argument(
        ...,
        help="Parameters to explore (e.g., Omega_bc w)",
        autocompletion=lambda: PARAM_CHOICES,
    ),
    resolution: int = Option(
        50, "--resolution", help="Number of grid points per dimension"
    ),
    cosmology: str = COSMOLOGY_OPTION,
    prior_names: List[str] = PRIORS_OPTION,
    label: str = Option("", "--label", "-l", help="Label for the resulting contour"),
    fix: List[click.Tuple] = FIX_OPTION,
    var_range: List[click.Tuple] = RANGE_OPTION,
    free: List[str] = FREE_OPTION,
    confidence_threshold: float = Option(
        95.2,
        "--confidence-threshold",
        "-T",
        help="Maximal level of confidence in percent",
    ),
    mu: Optional[str] = MU_OPTION,
    mucov: Optional[str] = MU_COV_OPTION,
    output: str = Option(
        ...,
        "--output",
        "-o",
        help="Output file for contour data (e.g., contour_planck.pkl)",
    ),
):
    """Build 1D or 2D frequentists confidence maps"""
    from cosmologix import contours, tools

    priors = [get_prior(p) for p in prior_names] + load_mu(mu, mucov)
    fixed = parameters.Planck18.copy()
    to_free = parameters.DEFAULT_FREE[cosmology].copy()
    for par in to_free + free:
        fixed.pop(par)
    for par, value in fix:
        fixed[par] = value
    range_dict = tuple_list_to_dict(var_range)
    range_x = range_dict.get(params[0], parameters.DEFAULT_RANGE[params[0]])
    grid_params = {params[0]: range_x + [resolution]}
    if len(params) == 2:
        range_y = range_dict.get(params[1], parameters.DEFAULT_RANGE[params[1]])
        grid_params[params[1]] = range_y + [resolution]
        grid = contours.frequentist_contour_2d_sparse(
            priors,
            grid=grid_params,
            fixed=fixed,
            confidence_threshold=confidence_threshold,
        )
    elif len(params) == 1:
        grid = contours.frequentist_1d_profile(
            priors,
            grid=grid_params,
            fixed=fixed,
        )
    else:
        grid = {"list": []}
        for i, param1 in enumerate(params):
            grid_params = {param1: parameters.DEFAULT_RANGE[param1] + [resolution]}
            grid["list"].append(
                contours.frequentist_1d_profile(
                    priors,
                    grid=grid_params,
                    fixed=fixed,
                )
            )
            for param2 in params[i + 1 :]:
                grid_params = {
                    param1: range_dict.get(param1, parameters.DEFAULT_RANGE[param1])
                    + [resolution],
                    param2: range_dict.get(param2, parameters.DEFAULT_RANGE[param2])
                    + [resolution],
                }
                grid["list"].append(
                    contours.frequentist_contour_2d_sparse(
                        priors,
                        grid=grid_params,
                        fixed=fixed,
                        confidence_threshold=confidence_threshold,
                    )
                )
    if label:
        grid["label"] = label
    else:
        # Default label according to prior selection
        grid["label"] = "+".join(prior_names) + ("+SN" if mu is not None else "")
    tools.save(grid, output)
    print(f"Contour data saved to {output}")


@app.command()
def contour(
    input_files: List[str] = Argument(
        ..., help="Input file from explore (e.g., contour_planck.pkl)"
    ),
    output: Optional[str] = Option(
        None, "--output", "-o", help="Output file for contour plot"
    ),
    not_filled: List[int] = Option(
        [], "--not-filled", help="Use line contours at INDEX (e.g., --not-filled 0)"
    ),
    color: List[str] = Option(
        [], "--color", help="Color for contour at INDEX (e.g., --color 0 red)"
    ),
    label: List[str] = Option(
        [], "--label", help="Override label for contour at INDEX (e.g., --label 0 CMB)"
    ),
    levels: List[float] = Option(
        [68.0, 95.0], "--levels", help="Contour levels (e.g., --levels 68 95)"
    ),
    legend_loc: str = Option(
        "best",
        "--legend-loc",
        help="Legend location",
        show_choices=True,
        autocompletion=lambda: [
            "best",
            "upper left",
            "upper right",
            "lower left",
            "lower right",
            "center",
            "outside",
        ],
    ),
    contour_index: int = Option(
        0, help="Index of the contour for files with multiple contours"
    ),
    show: bool = Option(False, "--show", "-s", help="Display the contour plot"),
    latex: bool = Option(
        False, "--latex", "-l", help="Plot in paper format using LaTeX"
    ),
):
    """Display (or save) a contour plot from explore output."""
    from cosmologix import tools, display
    import matplotlib.pyplot as plt

    if len(color) % 2 != 0 or len(label) % 2 != 0:
        raise typer.BadParameter("--color and --label require pairs of INDEX and VALUE")
    color_pairs = {int(color[i]): color[i + 1] for i in range(0, len(color), 2)}
    label_pairs = {int(label[i]): label[i + 1] for i in range(0, len(label), 2)}
    if latex:
        plt.rc("text", usetex=True)
        plt.rc("axes.spines", top=False, right=False)
    plt.figure()
    for i, input_file in enumerate(input_files):
        grid = tools.load(input_file)
        if "list" in grid:
            grid["list"][contour_index]["label"] = grid["label"]
            grid = grid["list"][contour_index]
        color = color_pairs.get(i, display.color_theme[i])
        label = label_pairs.get(i, grid["label"])
        if len(grid["params"]) == 2:
            display.plot_contours(
                grid,
                filled=i not in not_filled,
                color=color,
                label=label,
                levels=levels,
            )
        else:
            display.plot_profile(
                grid,
                color=color,
            )
    plt.legend(loc=legend_loc, frameon=False)
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=300)
        print(f"Contour plot saved to {output}")
        if show:
            plt.show()
    else:
        plt.show()
    plt.close()


@app.command()
def clear_cache():
    """Clear precompiled likelihoods."""
    from cosmologix import tools

    tools.clear_cache()


@app.command()
def corner(
    input_files: List[str] = Argument(
        ..., help="Input file from explore (e.g., contour_planck.pkl)"
    ),
    label: List[str] = Option(
        [], "--label", help="Override labels for contours (e.g., --labels 1 DR2)"
    ),
    output: Optional[str] = Option(
        None, "--output", "-o", help="Output file for corner plot"
    ),
    show: bool = Option(False, "--show", "-s", help="Display the corner plot"),
):
    """Produce a corner plot for a set of results."""
    from cosmologix import display, tools
    import matplotlib.pyplot as plt
    import jax.numpy as jnp

    axes = None
    param_names = None
    label_pairs = {int(label[i]): label[i + 1] for i in range(0, len(label), 2)}
    
    #confidence_contours = []
    for i, input_file in enumerate(input_files):
        result = tools.load(input_file)
        # distinguish between fit results and chi2 maps
        if "list" in result:
            axes, param_names = display.corner_plot_contours(
                result["list"],
                axes=axes,
                param_names=param_names,
                color=display.color_theme[i],
            )
            if i not in label_pairs:
                label_pairs[i] = result["label"]
        elif "params" not in result:
            axes, param_names = display.corner_plot_fisher(
                result, axes=axes, param_names=param_names, color=display.color_theme[i]
            )
            if i not in label_pairs:
                label_pairs[i] = result["label"] + " (Fisher)"

    # Disabling for now the possibility to add individual contours
    #    else:
    #        confidence_contours.append(result)
    #if confidence_contours:
    #    axes, param_names = display.corner_plot_contours(
    #        confidence_contours,
    #        axes=axes,
    #        param_names=param_names,
    #        color=display.color_theme[i],
    #    )
    for i, label in label_pairs.items():
        axes[0, -1].plot(jnp.nan, jnp.nan, color=display.color_theme[i], label=label)
    axes[0, -1].legend(frameon=True)
    axes[0, -1].set_visible(True)
    axes[0, -1].axis("off")
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=300)
        print(f"Corner plot saved to {output}")
        if show:
            plt.show()
    else:
        plt.show()


def main():
    """Cosmologix Command Line Interface."""
    app()


if __name__ == "__main__":
    main()
