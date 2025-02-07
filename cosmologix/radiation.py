"""Energy density of neutrinos and radiation

See Planck 2013 results XVI sect. 2.1.1 and Camb notes
downloadable at http://camb.info/readme.html.
See https://arxiv.org/pdf/astro-ph/0203507

Assumptions in code below are the following:
- Fermi-Dirac distribution for neutrinos
- decoupling while relativistic
- three active neutrinos
if mnu is non zero, a single eigenstate of mass mnu is
considered.

share_delta_neff = T
"""

import jax.numpy as jnp
from typing import Dict
from .tools import (
    Constants,
    safe_vmap,
    trapezoidal_rule_integration,
    linear_interpolation,
)


def rhoc(H: float) -> float:
    """Calculate the critical density in kg/m^3."""
    return 3 * (H * 1e-3 / Constants.pc) ** 2 / (8 * jnp.pi * Constants.G)


def Tcmb_to_Omega_gamma(Tcmb: float, H0: float) -> float:
    """Convert CMB temperature to Omega_gamma."""
    rhogamma = 4 * Constants.sigma * Tcmb**4 / Constants.c**3  # Energy density of CMB
    return rhogamma / rhoc(H0)


def _nu_dlnam() -> float:
    """Return the logarithmic step for neutrino mass scaling."""
    return -(jnp.log(Constants.am_min / Constants.am_max)) / (Constants.N - 1)


def Omega_nu_rel_per_nu(Omega_gamma: float) -> float:
    """Calculate the relativistic neutrino density per neutrino species."""
    return 7.0 / 8 * (4.0 / 11) ** (4.0 / 3) * Omega_gamma


def Omega_nu_h2(params: Dict[str, float]) -> float:
    """Compute the neutrino density parameter times h^2."""
    omgh2 = (
        Tcmb_to_Omega_gamma(params["Tcmb"], params["H0"]) * (params["H0"] / 100) ** 2
    )
    neutrino_mass_frac = 1 / (
        omgh2
        * (45 * Constants.zeta3)
        / (2 * jnp.pi**4)
        * (Constants.e / (Constants.k * params["Tcmb"]))
        * (4.0 / 11.0)
    )
    return params["m_nu"] / neutrino_mass_frac * (params["Neff"] / 3)


def Omega_n(params: Dict[str, float], u: float) -> float:
    """Compute the reduced density contribution of neutrinos at scale factor u."""
    return Omega_n_rel(params) + Omega_n_mass(params, u)


def Omega_n_rel(params: Dict[str, float]) -> float:
    """Calculate the reduced density contribution of relativistic neutrinos."""
    omg = Tcmb_to_Omega_gamma(params["Tcmb"], params["H0"])
    omr = Omega_nu_rel_per_nu(omg)
    omrnomass = omr * (
        params["Neff"] - 3.046 / 3.0
    )  # Note: This might need adjustment if massless neutrinos are considered
    return omrnomass


@safe_vmap(in_axes=(None, 0))
def Omega_n_mass(params: Dict[str, float], u: jnp.ndarray) -> jnp.ndarray:
    """Compute the reduced density contribution of massive neutrinos at scale factor u."""
    Omega_nu = Omega_nu_h2(params) / ((params["H0"] / 100) ** 2)
    Omega_massless_per_nu = Omega_nu_rel_per_nu(
        Tcmb_to_Omega_gamma(params["Tcmb"], params["H0"])
    )
    Nu_mass_fractions = 1
    Nu_mass_degeneracies = params["Neff"] / 3.0
    rho_massless_over_n = Constants.const / (1.5 * Constants.zeta3)
    nu_mass = (
        (Nu_mass_fractions / Nu_mass_degeneracies)
        * rho_massless_over_n
        * Omega_nu
        / Omega_massless_per_nu
    )
    return Omega_massless_per_nu * nu_rho(u**2 * nu_mass)


def nu_rho(am: float) -> float:
    """Calculate the density of one eigenstate of massive neutrinos."""
    adq = Constants.qmax / float(Constants.nq)
    q = jnp.arange(adq, Constants.qmax + adq, adq)
    aq = am / q
    v = 1 / jnp.sqrt(1 + aq**2)
    aqdn = q**3 / (1 + jnp.exp(q))
    drhonu = aqdn / v
    return jnp.sum(drhonu) * adq / Constants.const
