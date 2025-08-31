"""Python package for computing cosmological distances and parameters.

This package provides tools for cosmological parameter fitting in JAX.

Modules:
    cli: High-level functions for command-line interface.
    distances: Common cosmological functions.
    parameters: Default cosmological parameters.
    likelihoods: Observational constraints from various probes.
    fitter: Chi-squared minimization in JAX.
    contours: Frequentist confidence contours.
    display: Plotting tools.
"""

__all__ = [
    "cli",
    "distances",
    "parameters",
    "likelihoods",
    "fitter",
    "contours",
    "display",
]