#!/usr/bin/env python

import argparse
import pickle
from cosmologix import mu, fit, contours, likelihoods, Planck18
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Define available priors (extend this as needed)
AVAILABLE_PRIORS = {
    "Planck18": likelihoods.Planck2018Prior,
    "DESI2024": likelihoods.DESI2024Prior,
    "DES-5yr": likelihoods.DES5yr,
}

# Default fixed parameters for flat w-CDM
CMB_FREE = ['Omega_b_h2', 'H0']
DEFAULT_FREE = {
    'FLCDM': ['Omega_m'] + CMB_FREE,
    'FwCDM': ['Omega_m', 'w'] + CMB_FREE,
    'FwwaCDM': ['Omega_m', 'w', 'wa'] + CMB_FREE,
}

# Default ranges for the exploration of parameters
DEFAULT_RANGE = {
    'Omega_m': [0.18, 0.48],
    'w': [-0.6, -1.5],
    'wa': [-1, 1],
}

def main():
    parser = argparse.ArgumentParser(description="Cosmologix Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- 'fit' command ---
    fit_parser = subparsers.add_parser("fit", help="Fit a cosmological model to data")
    fit_parser.add_argument(
        "-p", "--priors", 
        nargs="*", 
        choices=AVAILABLE_PRIORS.keys(), 
        #required=True, 
        help="Priors to use (e.g., Planck18 DESI2024)"
    )
    fit_parser.add_argument(
        "-c", "--cosmology", 
        default="FwCDM", 
        choices=DEFAULT_FREE.keys(),  # Add more models as needed
        help="Cosmological model (default: FwCDM)"
    )
    fit_parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Output file for best-fit parameters (e.g., planck_desi.pkl)"
    )

    # --- 'explore' command ---
    explore_parser = subparsers.add_parser("explore", help="Explore a 2D parameter space")
    explore_parser.add_argument(
        "param1", 
        help="First parameter to explore (e.g., Omega_m)"
    )
    explore_parser.add_argument(
        "param2", 
        help="Second parameter to explore (e.g., w)"
    )
    explore_parser.add_argument(
        "--resolution", 
        type=int, 
        default=30, 
        help="Number of grid points per dimension (default: 30)"
    )
    explore_parser.add_argument(
        "-c", "--cosmology", 
        default="FwCDM", 
        choices=DEFAULT_FREE.keys(), 
        help="Cosmological model (default: FwCDM)"
    )
    explore_parser.add_argument(
        "-p", "--priors", 
        nargs="*", 
        choices=AVAILABLE_PRIORS.keys(), 
        required=True, 
        help="Priors to use (e.g., Planck18)"
    )
    explore_parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Output file for contour data (e.g., contour_planck.pkl)"
    )

    # --- 'contour' command ---
    contour_parser = subparsers.add_parser("contour", help="Plot a contour from explore output")
    contour_parser.add_argument(
        "input_files",
        nargs="+",
        help="Input file from explore (e.g., contour_planck.pkl)"
    )
    contour_parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Output file for contour plot (e.g., contour.png)"
    )

    args = parser.parse_args()

    if args.command == "fit":
        run_fit(args)
    elif args.command == "explore":
        run_explore(args)
    elif args.command == "contour":
        run_contour(args)
    else:
        parser.print_help()

def run_fit(args):
    """Fit the cosmological model and save the best-fit parameters."""
    priors = [AVAILABLE_PRIORS[p]() for p in args.priors]
    fixed = Planck18.copy()
    for par in DEFAULT_FREE[args.cosmology]:
        fixed.pop(par)
    result = fit(priors, fixed=fixed, verbose=True)
    print(result['bestfit'])
    with open(args.output, 'wb') as f:
        pickle.dump(result, f)
    print(f"Best-fit parameters saved to {args.output}")

def run_explore(args):
    """Explore a 2D parameter space and save the contour data."""
    priors = [AVAILABLE_PRIORS[p]() for p in args.priors]
    print(priors)
    grid_params = {
        args.param1: DEFAULT_RANGE[args.param1] + [args.resolution],
        args.param2: DEFAULT_RANGE[args.param2] + [args.resolution]
    }
    fixed = Planck18.copy()
    for par in DEFAULT_FREE[args.cosmology]:
        fixed.pop(par)
    grid = contours.frequentist_contour_2D_sparse(
        priors,
        grid=grid_params,
        fixed=fixed
    )
    contours.save_contours(grid, args.output)
    print(f"Contour data saved to {args.output}")

def run_contour(args):
    """Generate and save a contour plot from explore output."""
    plt.figure()
    for input_file in args.input_files:
        grid = contours.load_contours(input_file)
        contours.plot_contours(grid, filled=True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"Contour plot saved to {args.output}")

if __name__ == "__main__":
    main()
