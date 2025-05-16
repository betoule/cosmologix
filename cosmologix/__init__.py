"""Python package for computing cosmological distances, and
cosmological parameter fit in JAX.

Look into cosmologix.cli for high level functions.

The core library is organized as follows:
- distances: common cosmological functions
- parameters: default cosmological parameters
- likelihoods: observationnal constraints from various probes
- fitter: chi2 minimization in jax
- contours: frequentist confidence contours
- display: plotting tools
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
