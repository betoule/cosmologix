#!/usr/bin/env python

import argparse
import pickle
from cosmologix import mu, fit, contours, likelihoods, Planck18, fitter
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Define available priors (extend this as needed)
AVAILABLE_PRIORS = {
    "Planck18": likelihoods.Planck2018Prior,
    "DESI2024": likelihoods.DESI2024Prior,
    "DES-5yr": likelihoods.DES5yr,
}

# Default fixed parameters for flat w-CDM
CMB_FREE = ["Omega_b_h2", "H0"]
DEFAULT_FREE = {
    "FLCDM": ["Omega_m"] + CMB_FREE,
    "FwCDM": ["Omega_m", "w"] + CMB_FREE,
    "FwwaCDM": ["Omega_m", "w", "wa"] + CMB_FREE,
}

# Default ranges for the exploration of parameters
DEFAULT_RANGE = {
    "Omega_m": [0.18, 0.48],
    "w": [-0.6, -1.5],
    "wa": [-1, 1],
}


def pretty_print(result):
    """Pretty-print best-fit parameters with uncertainties from the Fisher Information Matrix."""
    bestfit = result["bestfit"]
    ifim = result["inverse_FIM"]  # Fisher Information Matrix (inverse covariance)

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


def main():
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
        metavar="PARAM",
        choices=list(Planck18.keys()),
        help="Fix the specified PARAM (e.g. -F H0 -F Omega_b_h2).",
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
        help="Attempt to impose hard priors on parameters not constrained by the selected dataset. Use with caution.",
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
        "param1", help="First parameter to explore (e.g., Omega_m)"
    )
    explore_parser.add_argument("param2", help="Second parameter to explore (e.g., w)")
    explore_parser.add_argument(
        "--resolution",
        type=int,
        default=30,
        help="Number of grid points per dimension (default: 30)",
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
        metavar="PARAM",
        choices=list(Planck18.keys()),
        help="Fix the specified PARAM (e.g. -F H0 -F Omega_b_h2).",
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
        help="Specify to use line contours instead of filled contour at INDEX (e.g, --not-filled 0)",
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

    args = parser.parse_args()

    if args.command == "fit":
        run_fit(args)
    elif args.command == "explore":
        run_explore(args)
    elif args.command == "contour":
        args.color = {int(index): color for index, color in args.color}
        args.label = {int(index): label for index, label in args.label}
        run_contour(args)
    else:
        parser.print_help()


def load_mu(args):
    import numpy as np

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
    else:
        return []


def auto_restricted_fit(priors, fixed, verbose):
    for retry in range(3):
        try:
            result = fit(priors, fixed=fixed, verbose=verbose)
            break
        except fitter.UnconstrainedParameterError as e:
            for param in e.params:
                print(f"Fixing unconstrained parameter {param[0]}")
                fixed[param[0]] = Planck18[param[0]]
        except fitter.DegenerateParametersError as e:
            print(f"Try again fixing H0")
            fixed["H0"] = Planck18["H0"]
    return result


def run_fit(args):
    """Fit the cosmological model and save the best-fit parameters."""
    priors = [AVAILABLE_PRIORS[p]() for p in args.priors] + load_mu(args)
    fixed = Planck18.copy()
    to_free = DEFAULT_FREE[args.cosmology].copy()
    for par in args.fix:
        if par in to_free:
            print(f"{par} kept fixed")
            to_free.remove(par)
    for par in to_free:
        fixed.pop(par)
    if args.auto_constrain:
        result = auto_restricted_fit(priors, fixed, args.verbose)
    else:
        result = fit(priors, fixed=fixed, verbose=args.verbose)
    pretty_print(result)
    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump(result, f)
        print(f"Best-fit parameters saved to {args.output}")


def run_explore(args):
    """Explore a 2D parameter space and save the contour data."""
    priors = [AVAILABLE_PRIORS[p]() for p in args.priors] + load_mu(args)
    print(priors)
    grid_params = {
        args.param1: DEFAULT_RANGE[args.param1] + [args.resolution],
        args.param2: DEFAULT_RANGE[args.param2] + [args.resolution],
    }
    fixed = Planck18.copy()
    to_free = DEFAULT_FREE[args.cosmology].copy()
    for par in args.fix:
        if par in to_free:
            print(f"{par} kept fixed")
            to_free.remove(par)
    for par in to_free:
        fixed.pop(par)
    grid = contours.frequentist_contour_2D_sparse(priors, grid=grid_params, fixed=fixed)
    if args.label:
        grid["label"] = args.label
    else:
        # Default label according to prior selection
        grid["label"] = "+".join(args.priors)
    contours.save_contours(grid, args.output)
    print(f"Contour data saved to {args.output}")


def run_contour(args):
    """Generate and save a contour plot from explore output."""
    if args.latex:
        plt.rc("text", usetex=True)
        plt.rc("axes.spines", top=False, right=False)  # , bottom=False, left=False)
    plt.figure()
    for i, input_file in enumerate(args.input_files):
        grid = contours.load_contours(input_file)
        base_color = args.color.get(i, contours.color_theme[i])
        label = args.label.get(i, None)
        contours.plot_contours(
            grid, filled=i not in args.not_filled, base_color=base_color, label=label
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
