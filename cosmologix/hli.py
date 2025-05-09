"""Cosmologix high level interface

A collection of high level functions for common tasks, also accessible
from the command line

"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from cosmologix import contours, likelihoods, fitter, display, tools, parameters


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
