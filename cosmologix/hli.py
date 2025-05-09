"""Cosmologix high level interface

A collection of high level functions for common tasks, also accessible
from the command line

"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from cosmologix import contours, likelihoods, fitter, display, tools, parameters

def run_fit(args):
    """Fit the cosmological model and save the best-fit parameters."""


def run_explore(args):
    """Explore a 2D parameter space and save the contour data."""
    priors = [get_prior(p) for p in args.priors] + load_mu(args)
    print(priors)
    fixed = parameters.Planck18.copy()
    to_free = parameters.DEFAULT_FREE[args.cosmology].copy()
    for par in to_free + args.free:
        fixed.pop(par)
    for par, value in args.fix:
        fixed[par] = value
    range_x = (
        args.range_x
        if args.range_x is not None
        else parameters.DEFAULT_RANGE[args.params[0]]
    )
    grid_params = {args.params[0]: range_x + [args.resolution]}
    if len(args.params) == 2:
        range_y = (
            args.range_y
            if args.range_y is not None
            else parameters.DEFAULT_RANGE[args.params[1]]
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
            grid_params = {param1: parameters.DEFAULT_RANGE[param1] + [args.resolution]}
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
                    param1: parameters.DEFAULT_RANGE[param1] + [args.resolution],
                    param2: parameters.DEFAULT_RANGE[param2] + [args.resolution],
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
        color = args.color.get(i, display.color_theme[i])
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
