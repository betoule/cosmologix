"""
Fitting formulae for the acoustic scale
"""

import jax.numpy as jnp
from typing import Callable, Tuple, Dict
from .tools import Constants
from .distances import Omega_c, Omega_de, dM
from .radiation import Omega_n_mass, Omega_n_rel, Tcmb_to_Omega_gamma


#
# Approximation for z_star and z_drag
#
def z_star(params):
    """Redshift of the recombination"""
    Obh2 = params["Omega_b_h2"]
    h2 = params["H0"] ** 2 * 1e-4
    odm = params["Omega_m"]
    g1 = 0.0783 * Obh2**-0.238 / (1 + 39.5 * Obh2**0.763)
    g2 = 0.560 / (1 + 21.1 * Obh2**1.81)
    return 1048 * (1 + 0.00124 * Obh2**-0.738) * (1 + g1 * (odm * h2) ** g2)


def z_drag(params):
    """Redshift of the drag epoch

    Fitting formulae for adiabatic cold dark matter cosmology.
    Eisenstein & Hu (1997) Eq.4, ApJ 496:605
    """
    omegamh2 = params["Omega_m"] * (params["H0"] * 1e-2) ** 2
    b1 = 0.313 * (omegamh2**-0.419) * (1 + 0.607 * omegamh2**0.674)
    b2 = 0.238 * omegamh2**0.223
    return (
        1291
        * (1.0 + b1 * params["Omega_b_h2"] ** b2)
        * omegamh2**0.251
        / (1 + 0.659 * omegamh2**0.828)
    )


def a4H2(params, a):
    """Return a**4 * H(a)**2/H0**2"""
    h = params["H0"] / 100
    Omega_b0 = params["Omega_b_h2"] / h**2
    Omega_c0 = Omega_c(params)
    # Omega_nu_mass = jnp.array([Omega_n_mass(params, jnp.sqrt(aa)) for aa in a])
    Omega_nu_mass = Omega_n_mass(params, jnp.sqrt(a))
    Omega_nu_rel = Omega_n_rel(params)
    Omega_de0 = Omega_de(params, Omega_nu_rel)
    Omega_gamma = Tcmb_to_Omega_gamma(params["Tcmb"], params["H0"])
    return (
        (Omega_b0 + Omega_c0) * a
        + params["Omega_k"] * (a**2)
        + Omega_gamma
        + Omega_nu_rel
        + Omega_nu_mass
        + Omega_de0 * a ** (1 - 3.0 * params["w"])
    )


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


def rs(params, z):
    """The comoving sound horizon size in Mpc"""
    nstep = 1000
    a = 1.0 / (1.0 + z)
    _a = jnp.linspace(1e-8, a, nstep)
    _a = 0.5 * (_a[1:] + _a[:-1])
    step = _a[1] - _a[0]
    # step = a / nstep
    # _a = jnp.arange(0.5 * step, a, step)
    R = Constants.c * 1e-3 / jnp.sqrt(3) * dsound_da_approx(params, _a).sum() * step
    return R

def rd_approx(params):
    """
    The comoving sound horizon size at drag redshift in Mpc
    Formula from DESI 1yr cosmological result paper arxiv:2404.03002
    """
    omega_b = params["Omega_b_h2"]
    omega_m = params["Omega_m"] * (params["H0"]/100)**2
    return 147.05*(omega_m/0.1432)**(-0.23)*(params["Neff"]/3.04)**(-0.1)*(omega_b/0.02236)**-0.13


def theta_MC(params):
    """CosmoMC approximation of acoustic scale angle

    The code returns 100 θ_MC which is the sampling variable in Planck
    chains.
    """
    zstar = z_star(params)
    rsstar = rs(params, zstar)
    return rsstar / dM(params, zstar) * 100.0
