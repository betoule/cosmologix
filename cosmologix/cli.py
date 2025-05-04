#!/usr/bin/env python

"""
Cosmologix Command Line Interface
"""

import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt

from cosmologix import fit, contours, likelihoods, Planck18, fitter, display, tools

# Define available priors (extend this as needed)
AVAILABLE_PRIORS = {
    "Planck18": likelihoods.Planck2018Prior,
    "PR4": likelihoods.PR4,
    "DESIDR1": likelihoods.DESIDR1Prior,
    "DESIDR2": likelihoods.DESIDR2Prior,
    "DES-5yr": likelihoods.DES5yr,
    "Pantheon+": likelihoods.Pantheonplus,
    "Union3": likelihoods.Union3,
    "SH0ES": likelihoods.SH0ES,
}

# Default fixed parameters for flat w-CDM
CMB_FREE = ["Omega_b_h2", "H0"]
DEFAULT_FREE = {
    "FLCDM": ["Omega_bc"] + CMB_FREE,
    "LCDM": ["Omega_bc", "Omega_k"] + CMB_FREE,
    "FwCDM": ["Omega_bc", "w"] + CMB_FREE,
    "wCDM": ["Omega_bc", "Omega_k", "w"] + CMB_FREE,
    "FwwaCDM": ["Omega_bc", "w", "wa"] + CMB_FREE,
    "wwaCDM": ["Omega_bc", "Omega_k", "w", "wa"] + CMB_FREE,
}

# Default ranges for the exploration of parameters
DEFAULT_RANGE = {
    "Omega_bc": [0.18, 0.48],
    "Omega_k": [-0.3, 0.4],
    "w": [-0.6, -1.5],
    "wa": [-1, 1],
    "Omega_b_h2": [0.01, 0.04],
}


def validate_fix(*args):
    """string or float"""
    # pylint: disable=consider-iterating-dictionary
    if args[0] in list(Planck18.keys()) + ["M", "rd"]:
        return args[0]
    return float(args[0])


def main():
    """
    actual main
    """
    # pylint: disable=too-many-statements
    parser = argparse.ArgumentParser(description="Cosmologix Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- 'fit' command ---
    fit_parser = subparsers.add_parser("fit", help="Fit a cosmological model to data")
    fit_parser.add_argument(
        "-p",
        "--priors",
        nargs="*",
        choices=list(AVAILABLE_PRIORS.keys()),
        default=[],
        help="Priors to use (e.g., Planck18 DESI2024)",
    )
    fit_parser.add_argument(
        "-c",
        "--cosmology",
        default="FwCDM",
        choices=list(DEFAULT_FREE.keys()),  # Add more models as needed
        help="Cosmological model (default: FwCDM)",
    )
    fit_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Display the successive steps of the fit",
    )
    fit_parser.add_argument(
        "-F",
        "--fix",
        action="append",
        default=[],
        nargs=2,
        type=validate_fix,
        metavar="PARAM VALUE",
        help="Fix the specified PARAM at the specified value"
        " (e.g. -F H0 70 -F Omega_b_h2 0.02222).",
    )
    fit_parser.add_argument(
        "--free",
        action="append",
        default=[],
        metavar="PARAM",
        choices=list(Planck18.keys()) + ["M", "rd"],
        help="Force release of parameter PARAM (e.g. --free Neff).",
    )
    fit_parser.add_argument(
        "--mu",
        nargs="+",  # Accept 1 or 2 arguments
        help="Distance modulus data file and optional covariance matrix in npy format",
    )
    fit_parser.add_argument(
        "-A",
        "--auto-constrain",
        action="store_true",
        default=False,
        help="Attempt to impose hard priors on parameters not constrained by the selected dataset."
        " Use with caution.",
    )
    fit_parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        default=False,
        help="Display best-fit results as a corner plot.",
    )
    fit_parser.add_argument(
        "-o",
        "--output",
        help="Output file for best-fit parameters (e.g., planck_desi.pkl)",
    )

    # --- 'explore' command ---
    explore_parser = subparsers.add_parser(
        "explore", help="Explore a 2D parameter space"
    )
    explore_parser.add_argument(
        "params", nargs="+", help="parameters to explore (e.g., Omega_bc w)"
    )
    explore_parser.add_argument(
        "--resolution",
        type=int,
        default=50,
        help="Number of grid points per dimension (default: 50)",
    )
    explore_parser.add_argument(
        "-c",
        "--cosmology",
        default="FwCDM",
        choices=DEFAULT_FREE.keys(),
        help="Cosmological model (default: FwCDM)",
    )
    explore_parser.add_argument(
        "-p",
        "--priors",
        nargs="*",
        default=[],
        choices=AVAILABLE_PRIORS.keys(),
        help="Priors to use (e.g., Planck18)",
    )
    explore_parser.add_argument(
        "-l",
        "--label",
        default="",
        help="Gives a label to the resulting contour, to be latter used in plots (e.g., CMB+SN)",
    )
    explore_parser.add_argument(
        "-F",
        "--fix",
        action="append",
        default=[],
        nargs=2,
        type=validate_fix,
        metavar="PARAM VALUE",
        help="Fix the specified PARAM at the provided value (e.g. -F H0 70 -F Omega_b_h2 0.02222).",
    )
    explore_parser.add_argument(
        "--free",
        action="append",
        default=[],
        metavar="PARAM",
        choices=list(Planck18.keys()) + ["M", "rd"],
        help="Force release of parameter PARAM (e.g. --free Neff).",
    )
    explore_parser.add_argument(
        "-r",
        "--range-x",
        nargs=2,
        type=float,
        default=None,
        metavar="MIN MAX",
        help="Overide the exploration range for first parameter",
    )
    explore_parser.add_argument(
        "-R",
        "--range-y",
        nargs=2,
        type=float,
        default=None,
        metavar="MIN MAX",
        help="Overide the exploration range for second parameter",
    )
    explore_parser.add_argument(
        "-T",
        "--confidence-threshold",
        type=float,
        default=95.0,
        metavar="CONFIDENCE_LEVEL",
        help="Maximal level of confidence explored in percent (default 95%%).",
    )
    explore_parser.add_argument(
        "--mu",
        nargs="+",  # Accept 1 or 2 arguments
        help="Distance modulus data file and optional covariance matrix in npy format",
    )

    explore_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file for contour data (e.g., contour_planck.pkl)",
    )

    # --- 'contour' command ---
    contour_parser = subparsers.add_parser(
        "contour", help="Plot a contour from explore output"
    )
    contour_parser.add_argument(
        "input_files",
        nargs="+",
        help="Input file from explore (e.g., contour_planck.pkl)",
    )
    contour_parser.add_argument(
        "-o", "--output", help="Output file for contour plot (e.g., contour.png)"
    )
    contour_parser.add_argument(
        "--not-filled",
        action="append",
        type=int,
        metavar="INDEX",
        default=[],
        help="Specify to use line contours instead of filled contour at INDEX"
        " (e.g, --not-filled 0)",
    )
    contour_parser.add_argument(
        "--color",
        action="append",
        nargs=2,
        default=[],
        metavar=("INDEX", "COLOR"),
        help="Specify color for contour at INDEX (e.g., --color 0 red)",
    )
    contour_parser.add_argument(
        "--label",
        action="append",
        nargs=2,
        default=[],
        metavar=("INDEX", "LABEL"),
        help="Overide label for contour at INDEX (e.g., --label 0 CMB)",
    )
    contour_parser.add_argument(
        "--levels",
        nargs="+",
        default=[68.0, 95.0],
        type=float,
        metavar=("level1 level2 ..."),
        help="Plot the contours corresponding to the list of given levels (default: 68 95)",
    )
    contour_parser.add_argument(
        "--legend-loc",
        default="best",
        choices=[
            "best",
            "upper left",
            "upper right",
            "lower left",
            "lower right",
            "center",
            "outside",
        ],
        help="Legend location (default: 'best')",
    )
    contour_parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        default=False,
        help="Display the contour plot even if the plot is saved (default False)",
    )
    contour_parser.add_argument(
        "-l",
        "--latex",
        action="store_true",
        default=False,
        help="Plot in paper format, using latex for the text.",
    )
    clear_parser = subparsers.add_parser(
        "clear_cache", help="Clear precompiled likelihoods"
    )
    corner_parser = subparsers.add_parser(
        "corner", help="Produce a corner plot for a set of results"
    )
    corner_parser.add_argument(
        "input_files",
        nargs="+",
        help="Input file from explore (e.g., contour_planck.pkl)",
    )
    corner_parser.add_argument(
        "--labels",
        nargs="+",
        default=[],
        help="Labels for the contours corresponding to provided files (e.g., DR1 DR2)",
    )
    corner_parser.add_argument(
        "-o", "--output", help="Output file for corner plot (e.g., corner.png)"
    )
    corner_parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        default=False,
        help="Display the corner plot even if the plot is saved (default False)",
    )

    args = parser.parse_args()

    if args.command == "fit":
        run_fit(args)
    elif args.command == "explore":
        run_explore(args)
    elif args.command == "contour":
        args.color = {int(index): color for index, color in args.color}
        args.label = {int(index): label for index, label in args.label}
        run_contour(args)
    elif args.command == "clear_cache":
        tools.clear_cache()
    elif args.command == "corner":
        run_corner(args)
    else:
        parser.print_help()


def load_mu(args):
    """Load distance measurement."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    if args.mu:
        muobs = np.load(args.mu[0])
        if len(args.mu) == 2:
            cov = np.load(args.mu[1])
            like = likelihoods.MuMeasurements(muobs["z"], muobs["mu"], cov)
        else:
            like = likelihoods.DiagMuMeasurements(
                muobs["z"], muobs["mu"], muobs["muerr"]
            )
        return [like]
    return []


def auto_restricted_fit(priors, fixed, verbose):
    """Test if there is unconstrained parameters"""
    for _ in range(3):
        try:
            result = fit(priors, fixed=fixed, verbose=verbose)
            break
        except fitter.UnconstrainedParameterError as e:
            for param in e.params:
                print(f"Fixing unconstrained parameter {param[0]}")
                fixed[param[0]] = Planck18[param[0]]
        except fitter.DegenerateParametersError:
            print("Try again fixing H0")
            fixed["H0"] = Planck18["H0"]
    return result


def run_fit(args):
    """Fit the cosmological model and save the best-fit parameters."""
    priors = [AVAILABLE_PRIORS[p]() for p in args.priors] + load_mu(args)
    fixed = Planck18.copy()
    to_free = DEFAULT_FREE[args.cosmology].copy()
    for par in to_free + args.free:
        fixed.pop(par)
    for par, value in args.fix:
        fixed[par] = value
    if args.auto_constrain:
        result = auto_restricted_fit(priors, fixed, args.verbose)
    else:
        result = fit(priors, fixed=fixed, verbose=args.verbose)
    display.pretty_print(result)
    if args.output:
        tools.save(result, args.output)
        print(f"Best-fit parameters saved to {args.output}")
    if args.show:
        display.corner_plot_fisher(result)
        plt.tight_layout()
        plt.show()


def run_explore(args):
    """Explore a 2D parameter space and save the contour data."""
    priors = [AVAILABLE_PRIORS[p]() for p in args.priors] + load_mu(args)
    print(priors)
    fixed = Planck18.copy()
    to_free = DEFAULT_FREE[args.cosmology].copy()
    for par in to_free + args.free:
        fixed.pop(par)
    for par, value in args.fix:
        fixed[par] = value
    range_x = (
        args.range_x if args.range_x is not None else DEFAULT_RANGE[args.params[0]]
    )
    grid_params = {args.params[0]: range_x + [args.resolution]}
    if len(args.params) == 2:
        range_y = (
            args.range_y if args.range_y is not None else DEFAULT_RANGE[args.params[1]]
        )
        grid_params[args.params[1]] = range_y + [args.resolution]

        grid = contours.frequentist_contour_2d_sparse(
            priors,
            grid=grid_params,
            fixed=fixed,
            confidence_threshold=args.confidence_threshold,
        )
    elif len(args.params) == 1:
        grid = contours.frequentist_1d_profile(
            priors,
            grid=grid_params,
            fixed=fixed,
            # confidence_threshold=args.confidence_threshold,
        )
    else:
        grid = {"list": []}
        for i, param1 in enumerate(args.params):
            grid_params = {param1: DEFAULT_RANGE[param1] + [args.resolution]}
            grid["list"].append(
                contours.frequentist_1d_profile(
                    priors,
                    grid=grid_params,
                    fixed=fixed,
                    # confidence_threshold=args.confidence_threshold,
                )
            )
            for param2 in args.params[i + 1 :]:
                grid_params = {
                    param1: DEFAULT_RANGE[param1] + [args.resolution],
                    param2: DEFAULT_RANGE[param2] + [args.resolution],
                }
                grid["list"].append(
                    contours.frequentist_contour_2d_sparse(
                        priors,
                        grid=grid_params,
                        fixed=fixed,
                        confidence_threshold=args.confidence_threshold,
                    )
                )
    if args.label:
        grid["label"] = args.label
    else:
        # Default label according to prior selection
        grid["label"] = "+".join(args.priors)
    tools.save(grid, args.output)
    print(f"Contour data saved to {args.output}")


def run_contour(args):
    """Generate and save a contour plot from explore output."""
    if args.latex:
        plt.rc("text", usetex=True)
        plt.rc("axes.spines", top=False, right=False)  # , bottom=False, left=False)
    plt.figure()
    for i, input_file in enumerate(args.input_files):
        grid = tools.load(input_file)
        color = args.color.get(i, contours.color_theme[i])
        label = args.label.get(i, None)
        if len(grid["params"]) == 2:
            display.plot_contours(
                grid,
                filled=i not in args.not_filled,
                color=color,
                label=label,
                levels=args.levels,
            )
        else:
            display.plot_profile(
                grid,
                color=color,
            )
    plt.legend(loc=args.legend_loc, frameon=False)
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=300)
        print(f"Contour plot saved to {args.output}")
        if args.show:
            plt.show()
    else:
        plt.show()
    plt.close()


def run_corner(args):
    """make the corner plot."""
    axes = None
    param_names = None
    confidence_contours = []
    for i, input_file in enumerate(args.input_files):
        result = tools.load(input_file)
        # distinguish between fit results and chi2 maps
        if "list" in result:
            axes, param_names = display.corner_plot_contours(
                result["list"],
                axes=axes,
                param_names=param_names,
                color=display.color_theme[i],
            )
        elif "params" not in result:
            axes, param_names = display.corner_plot_fisher(
                result, axes=axes, param_names=param_names, color=display.color_theme[i]
            )
        else:
            confidence_contours.append(result)
    if confidence_contours:
        axes, param_names = display.corner_plot_contours(
            confidence_contours,
            axes=axes,
            param_names=param_names,
            color=display.color_theme[i],
        )
    for i, label in enumerate(args.labels):
        axes[0, -1].plot(jnp.nan, jnp.nan, color=display.color_theme[i], label=label)
    axes[0, -1].legend(frameon=True)
    axes[0, -1].set_visible(True)
    axes[0, -1].axis("off")
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=300)
        print(f"Corner plot saved to {args.output}")
        if args.show:
            plt.show()
    else:
        plt.show()


if __name__ == "__main__":
    main()
