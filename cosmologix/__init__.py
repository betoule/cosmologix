__all__ = ["distances", "likelihoods", "fitter"]

from .distances import mu
from .parameters import Planck18
from .fitter import fit


def lcdm_deviation(**keys):
    params = Planck18.copy()
    params.update(keys)
    return params
