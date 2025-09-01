#!/usr/bin/env python
"""Cosmologix high level interface.

A collection of script for common tasks also made available from the
command line thanks to typer.

"""

from typing import List, Optional, Annotated
import click
from typer import Typer, Option, Argument
from cosmologix import parameters, cli_tools

# We defer other imports to improve responsiveness on the command line
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-arguments,too-many-positional-arguments),too-many-locals

# Main Typer app
app = Typer(
    name="cosmologix",
    help="Cosmological fitting tool",
    no_args_is_help=True,
    add_completion=True,  # Enable default typer completion
)


def auto_restricted_fit(priors, fixed, verbose):
    """Tests for and fixes unconstrained parameters in a fit.

    Args:
        priors (list): A list of likelihood priors.
        fixed (dict): A dictionary of fixed parameters.
        verbose (bool): Whether to print verbose output.

    Returns:
        dict: The fit result.
    """
    from . import fitter

    default_params = parameters.get_cosmo_params()
    for _ in range(3):
        try:
            result = fitter.fit(priors, fixed=fixed, verbose=verbose)
            break
        except fitter.UnconstrainedParameterError as e:
            for param in e.params:
                print(f"Fixing unconstrained parameter {param[0]}")
                fixed[param[0]] = default_params[param[0]]
        except fitter.DegenerateParametersError:
            print("Try again fixing H0")
            fixed["H0"] = default_params["H0"]
    return result


@app.command()
def fit(
    prior_names: Annotated[Optional[List[str]], cli_tools.PRIORS_OPTION] = None,
    cosmology: Annotated[str, cli_tools.COSMOLOGY_OPTION] = "FwCDM",
    verbose: Annotated[
        bool, Option("--verbose", "-v", help="Display the successive steps of the fit")
    ] = False,
    fix: Annotated[Optional[List[click.Tuple]], cli_tools.FIX_OPTION] = None,
    free: Annotated[Optional[List[str]], cli_tools.FREE_OPTION] = None,
    mu: Annotated[Optional[str], cli_tools.MU_OPTION] = None,
    mucov: Annotated[Optional[str], cli_tools.MU_COV_OPTION] = None,
    auto_constrain: Annotated[
        bool,
        Option(
            "--auto-constrain",
            "-A",
            help="Impose hard priors on unconstrained parameters (use with caution)",
        ),
    ] = False,
    show: Annotated[
        bool, Option("--show", "-s", help="Display best-fit results as a corner plot")
    ] = False,
    strict_fom: Annotated[
        bool,
        Option(
            "--strict-fom",
            help="Report DETF FoM as reciprocal area of the 95% contour instead of the more commonly used 1/(sigma(w_p)sigma(w_a)).",
        ),
    ] = False,
    output: Annotated[
        Optional[str],
        Option(
            "--output",
            "-o",
            help="Output file for best-fit parameters (e.g., planck_desi.pkl)",
        ),
    ] = None,
):
    """Finds the best-fit cosmological model."""
    from . import fitter, display, tools

    fix = fix or []
    free = free or []
    prior_names = prior_names or []
    priors = [cli_tools.get_prior(p) for p in prior_names] + cli_tools.load_mu(
        mu, mucov
    )
    fixed = parameters.get_cosmo_params()
    to_free = parameters.DEFAULT_FREE[cosmology].copy()
    for par in to_free + free:
        fixed.pop(par)
    for par, value in fix:
        fixed[par] = value
    if auto_constrain:
        result = auto_restricted_fit(priors, fixed, verbose)
    else:
        result = fitter.fit(priors, fixed=fixed, verbose=verbose)
    if strict_fom:
        display.pretty_print(result, fom_scale=95)
    else:
        display.pretty_print(result, fom_scale="1sigma")
    result["label"] = "+".join(prior_names) + ("+SN" if mu is not None else "")
    if output:
        tools.save(result, output)
        print(f"Best-fit parameters saved to {output}")
    if show:
        import matplotlib.pyplot as plt

        display.corner_plot_fisher(result)
        plt.tight_layout()
        plt.show()
    return result


@app.command()
def explore(
    params: Annotated[
        List[str],
        Argument(
            help="Parameters to explore (e.g., Omega_bc w)",
            autocompletion=lambda: cli_tools.PARAM_CHOICES,
        ),
    ],
    prior_names: Annotated[Optional[List[str]], cli_tools.PRIORS_OPTION] = None,
    resolution: Annotated[
        int, Option("--resolution", help="Number of grid points per dimension")
    ] = 50,
    cosmology: Annotated[str, cli_tools.COSMOLOGY_OPTION] = "FwCDM",
    label: Annotated[
        str, Option("--label", "-l", help="Label for the resulting contour")
    ] = "",
    fix: Annotated[Optional[List[click.Tuple]], cli_tools.FIX_OPTION] = None,
    free: Annotated[Optional[List[str]], cli_tools.FREE_OPTION] = None,
    var_range: Annotated[Optional[List[click.Tuple]], cli_tools.RANGE_OPTION] = None,
    confidence_threshold: Annotated[
        float,
        Option(
            "--confidence-threshold",
            "-T",
            help="Maximal explored level of confidence in percent",
        ),
    ] = 95.3,
    mu: Annotated[Optional[str], cli_tools.MU_OPTION] = None,
    mucov: Annotated[Optional[str], cli_tools.MU_COV_OPTION] = None,
    output: Annotated[
        str,
        Option(
            "--output",
            "-o",
            help="Output file for contour data (e.g., contour_planck.pkl)",
        ),
    ] = "",
):
    """Builds 1D or 2D frequentist confidence maps."""
    from cosmologix import contours, tools

    fix = fix or []
    free = free or []
    prior_names = prior_names or []
    var_range = var_range or []
    priors = [cli_tools.get_prior(p) for p in prior_names] + cli_tools.load_mu(
        mu, mucov
    )
    fixed = parameters.get_cosmo_params()
    to_free = parameters.DEFAULT_FREE[cosmology].copy()
    for par in to_free + free:
        fixed.pop(par)
    for par, value in fix:
        fixed[par] = value
    range_dict = cli_tools.tuple_list_to_dict(var_range)
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
            grid_params = {
                param1: range_dict.get(param1, parameters.DEFAULT_RANGE[param1])
                + [resolution]
            }
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
    if output:
        tools.save(grid, output)
        print(f"Contour data saved to {output}")
    return grid


@app.command()
def contour(
    input_files: Annotated[
        List[str], Argument(help="Input file from explore (e.g., contour_planck.pkl)")
    ],
    output: Annotated[
        Optional[str], Option("--output", "-o", help="Output file for contour plot")
    ] = None,
    not_filled: Annotated[
        Optional[List[int]],
        Option(
            "--not-filled", help="Use line contours at INDEX (e.g., --not-filled 0)"
        ),
    ] = None,
    colors: Annotated[Optional[List[click.Tuple]], cli_tools.COLORS_OPTION] = None,
    labels: Annotated[Optional[List[click.Tuple]], cli_tools.LABELS_OPTION] = None,
    levels: Annotated[
        Optional[List[float]],
        Option("--levels", help="Contour levels (e.g., --levels 68 95)"),
    ] = None,
    legend_loc: Annotated[
        str,
        Option(
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
    ] = "best",
    contour_index: Annotated[
        int, Option(help="Index of the contour for files with multiple contours")
    ] = 0,
    show: Annotated[
        bool, Option("--show", "-s", help="Display the contour plot")
    ] = False,
    latex: Annotated[
        bool, Option("--latex", help="Plot in paper format using LaTeX")
    ] = False,
):
    """Displays or saves a contour plot from the output of `explore`."""
    from cosmologix import tools, display
    import matplotlib.pyplot as plt

    levels = levels or [68.0, 95.0]
    not_filled = not_filled or []
    color_pairs = cli_tools.tuple_list_to_dict(colors or [])
    label_pairs = cli_tools.tuple_list_to_dict(labels or [])
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
                label=label,
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
def corner(
    input_files: Annotated[
        List[str], Argument(help="Input file from explore (e.g., contour_planck.pkl)")
    ],
    labels: Annotated[Optional[List[click.Tuple]], cli_tools.LABELS_OPTION] = None,
    colors: Annotated[Optional[List[click.Tuple]], cli_tools.COLORS_OPTION] = None,
    not_filled: Annotated[
        Optional[List[int]],
        Option(
            "--not-filled", help="Use line contours at INDEX (e.g., --not-filled 0)"
        ),
    ] = None,
    output: Annotated[
        Optional[str], Option("--output", "-o", help="Output file for corner plot")
    ] = None,
    show: Annotated[
        bool, Option("--show", "-s", help="Display the corner plot")
    ] = False,
    latex: Annotated[
        bool, Option("--latex", help="Plot in paper format using LaTeX")
    ] = False,
):
    """Produces a corner plot for a set of results."""
    from cosmologix import display, tools
    import matplotlib.pyplot as plt
    import jax.numpy as jnp

    axes = None
    param_names = None
    not_filled = not_filled or []
    label_pairs = cli_tools.tuple_list_to_dict(labels or [])
    color_pairs = cli_tools.tuple_list_to_dict(colors or [])
    if latex:
        plt.rc("text", usetex=True)
        plt.rc("axes.spines", top=False, right=False)
    # confidence_contours = []
    for i, input_file in enumerate(input_files):
        result = tools.load(input_file)
        # distinguish between fit results and chi2 maps
        if "list" in result:
            axes, param_names = display.corner_plot_contours(
                result["list"],
                axes=axes,
                param_names=param_names,
                color=color_pairs.get(i, display.color_theme[i]),
                filled=i not in not_filled,
            )
            if i not in label_pairs:
                label_pairs[i] = result["label"]
        elif "params" not in result:
            axes, param_names = display.corner_plot_fisher(
                result,
                axes=axes,
                param_names=param_names,
                color=color_pairs.get(i, display.color_theme[i]),
            )
            if i not in label_pairs:
                label_pairs[i] = result["label"] + " (Fisher)"

    # Disabling for now the possibility to add individual contours
    #    else:
    #        confidence_contours.append(result)
    # if confidence_contours:
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


@app.command()
def clear_cache(
    jit: Annotated[
        bool, Option("--jit", "-j", help="Clear only the persistent jit cache")
    ] = False,
):
    """Clears cached data."""
    from cosmologix import tools

    tools.clear_cache(jit)


def main():
    """The main entry point for the Cosmologix command-line interface."""
    app()


if __name__ == "__main__":
    main()
