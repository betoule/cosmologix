#!/usr/bin/env python

"""
Cosmologix Command Line Interface
"""

import argparse
from cosmologix import parameters

# Define available priors (extend this as needed)
# AVAILABLE_PRIORS = {
#    "Planck18": likelihoods.Planck2018Prior,
#    "PR4": likelihoods.PR4,
#    "DESIDR1": likelihoods.DESIDR1Prior,
#    "DESIDR2": likelihoods.DESIDR2Prior,
#    "DES-5yr": likelihoods.DES5yr,
#    "Pantheon+": likelihoods.Pantheonplus,
#    "Union3": likelihoods.Union3,
#    "SH0ES": likelihoods.SH0ES,
# }
# Define available priors (extend this as needed)
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


def validate_fix(*args):
    """string or float"""
    # pylint: disable=consider-iterating-dictionary
    if args[0] in list(parameters.Planck18.keys()) + ["M", "rd"]:
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
        choices=AVAILABLE_PRIORS,
        default=[],
        help="Priors to use (e.g., Planck18 DESI2024)",
    )
    fit_parser.add_argument(
        "-c",
        "--cosmology",
        default="FwCDM",
        choices=list(parameters.DEFAULT_FREE.keys()),  # Add more models as needed
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
        choices=list(parameters.Planck18.keys()) + ["M", "rd"],
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
        choices=parameters.DEFAULT_FREE.keys(),
        help="Cosmological model (default: FwCDM)",
    )
    explore_parser.add_argument(
        "-p",
        "--priors",
        nargs="*",
        default=[],
        choices=AVAILABLE_PRIORS,
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
        choices=list(parameters.Planck18.keys()) + ["M", "rd"],
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

    from cosmologix import hli, tools

    if args.command == "fit":
        hli.run_fit(args)
    elif args.command == "explore":
        hli.run_explore(args)
    elif args.command == "contour":
        args.color = {int(index): color for index, color in args.color}
        args.label = {int(index): label for index, label in args.label}
        hli.run_contour(args)
    elif args.command == "clear_cache":
        tools.clear_cache()
    elif args.command == "corner":
        hli.run_corner(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
