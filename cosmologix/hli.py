"""Cosmologix high level interface

A collection of high level functions for common tasks, also accessible
from the command line

"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from cosmologix import contours, likelihoods, fitter, display, tools, parameters


def run_corner(args):
    """make the corner plot."""
