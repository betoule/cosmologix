"""
Fitting formulae for the acoustic scale
"""

import jax.numpy as jnp
from typing import Callable, Tuple, Dict
from .tools import Constants
from .distances import dM
from cosmologix import densities


#
# Approximation for z_star and z_drag
#
def z_star(params):
    """Redshift of the recombination
    From Hu & Sugiyama (1996) Eq. E-1 
    """
    Obh2 = params["Omega_b_h2"]
    h2 = params["H0"] ** 2 * 1e-4
    odm = params["Omega_bc"] + params["Omega_nu"]
    g1 = 0.0783 * Obh2**-0.238 / (1 + 39.5 * Obh2**0.763)
    g2 = 0.560 / (1 + 21.1 * Obh2**1.81)
    return 1048 * (1 + 0.00124 * Obh2**-0.738) * (1 + g1 * (odm * h2) ** g2)


def z_drag(params):
    """Redshift of the drag epoch

    Fitting formulae for adiabatic cold dark matter cosmology.
    Eisenstein & Hu (1997) Eq.4, ApJ 496:605
    """
    omegamh2 = params["Omega_bc"] * (params["H0"] * 1e-2) ** 2
    b1 = 0.313 * (omegamh2**-0.419) * (1 + 0.607 * omegamh2**0.674)
    b2 = 0.238 * omegamh2**0.223
    return (
        1291
        * (1.0 + b1 * params["Omega_b_h2"] ** b2)
        * omegamh2**0.251
        / (1 + 0.659 * omegamh2**0.828)
    )


def a4H2(params, a):
    z = 1 / a - 1
    return a**4 * densities.Omega(params, z)


def dsound_da_approx(params, a):
    """Approximate form of the sound horizon used by cosmomc for theta

    Notes
    -----

    This is to be used in comparison with values in the cosmomc chains

    """
    return 1.0 / (
        jnp.sqrt(a4H2(params, a) * (1.0 + 3e4 * params["Omega_b_h2"] * a))
        * params["H0"]
    )


def dsound_da(params, a):
    """The exact integrand in the computation of rs

    Notes
    -----
    see e.g. Komatsu et al. (2009) eq. 6
    """
    return 1.0 / (
        jnp.sqrt(
            a4H2(params, a)
            * (1.0 + 0.75 * (params["Omega_b"] / params["Omega_gamma"]) * a)
        )
        * params["H0"]
    )


def rs(params, z):
    """The comoving sound horizon size in Mpc"""
    nstep = 1000
    a = 1.0 / (1.0 + z)
    _a = jnp.linspace(1e-8, a, nstep)
    _a = 0.5 * (_a[1:] + _a[:-1])
    step = _a[1] - _a[0]
    R = Constants.c * 1e-3 / jnp.sqrt(3) * dsound_da(params, _a).sum() * step
    return R


def rs_approx(params, z):
    """The approximated comoving sound horizon size in Mpc

    Notes
    -----
    Uses dsound_da_approx which is the formula in use to compute 100θ_MC in cosmomc
    """
    nstep = 1000
    a = 1.0 / (1.0 + z)
    _a = jnp.linspace(1e-8, a, nstep)
    _a = 0.5 * (_a[1:] + _a[:-1])
    step = _a[1] - _a[0]
    R = Constants.c * 1e-3 / jnp.sqrt(3) * dsound_da_approx(params, _a).sum() * step
    return R


def rd(params):
    """
    The comoving sound horizon size at drag redshift in Mpc
    """
    par = densities.process_params(params)
    return rs(par, z_drag(par))


def rd_approx(params):
    """
    Fit formula for the comoving sound horizon size at drag redshift in Mpc
    Formula from DESI 1yr cosmological result paper arxiv:2404.03002
    """
    omega_b = params["Omega_b_h2"]
    omega_m = params["Omega_bc"] * (params["H0"] / 100) ** 2
    return (
        147.05
        * (omega_m / 0.1432) ** (-0.23)
        * (params["Neff"] / 3.04) ** (-0.1)
        * (omega_b / 0.02236) ** -0.13
    )


def theta_MC(params):
    """CosmoMC approximation of acoustic scale angle

    The code returns 100 θ_MC which is the sampling variable in Planck
    chains.
    """
    params = densities.process_params(params)
    zstar = z_star(params)
    rsstar = rs_approx(params, zstar)
    return rsstar / dM(params, zstar) * 100.0
