from .distances import mu
from .fitter import fit
from .likelihoods import Planck18


def lcdm_deviation(**keys):
    params = Planck18.copy()
    params.update(keys)
    return params
