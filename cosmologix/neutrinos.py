from cosmologix import densities
from cosmologix.tools import Constants, trapezoidal_rule_integration, safe_vmap
from cosmologix.interpolation import (
    chebyshev_nodes,
    newton_interp,
    newton_divided_differences,
)
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

def compute_cmb_photon_density(Tcmb):
    """
    Compute the energy density of CMB photons today in kg/m^3.

    Parameters:
    -----------
    Tcmb : float
        CMB temperature today in K.

    Returns:
    --------
    float
        Energy density of CMB photons in kg/m^3.
    """
    return 4 * Constants.sigma * Tcmb ** 4 / Constants.c**3


def compute_neutrino_temperature(Tcmb, Neff):
    """
    Calculate the neutrino distribution temperature today.

    Based on the decoupling model described in 2005NuPhB.729..221M.

    Parameters:
    -----------
    Tcmb: float
        CMB temperature today in K.
    Neff: float
        effective number of neutrino species.

    Returns:
    --------
    float
        neutrino temperature today in K.
    """
    return (4 / 11) ** (1.0 / 3) * (Neff / 3) ** (1.0 / 4) * Tcmb
    

def compute_relativistic_neutrino_density(params):
    """
    Compute the energy density of relativistic neutrinos.

    Parameters:
    -----------
    params : dict
        A dictionary containing cosmological parameters.

    Returns:
    --------
    float
        Energy density of relativistic neutrinos in kg/m^3.
    """
    return (
        7.0
        / 8
        * params["Neff"]
        / 3
        * (4 / 11) ** (4.0 / 3)
        * compute_cmb_photon_density(params['Tcmb'])
    )


@safe_vmap()
def compute_neutrino_density(params, z):
    """
    Calculate the energy density of neutrinos at redshift z.

    Parameters:
    -----------
    params : dict
        A dictionary containing cosmological parameters.
    z : float or array
        Redshift.

    Returns:
    --------
    float or array
        Neutrino energy density at redshift z.
    """
    return (
        compute_relativistic_neutrino_density(params)
        * 120
        / (7 * jnp.pi**4)
        * compute_composite_integral(params["m_nu_bar"] / (1 + z))
    )


@safe_vmap(in_axes=(0,))
def compute_fermion_distribution_integral(m_bar):
    r"""
    Helper function to compute the integral of the energy distribution of massive fermions.

    This function evaluates:
    \int_0^\inf x^3 \sqrt(1 + (m_bar/x)^2)/(e^x + 1) dx

    Parameters:
    -----------
    m_bar : float or array
        Reduced mass parameter \bar m.

    Returns:
    --------
    float or array
        The result of the integral.

    Notes:
    ------
    This function admits expansions for the non-relativistic and ultra-relativistic cases.
    It is also very smooth in between, allowing for significant speed-up by combining
    expansions with a precomputed polynomial interpolant (see `compute_composite_integral`).
    """

    def integrand(x):
        return x**3 * jnp.sqrt(1 + (m_bar / x) ** 2) / (1 + jnp.exp(x))

    return trapezoidal_rule_integration(integrand, 1e-3, 31, 10000)


def convert_mass_to_reduced_parameter(m_nu, T_nu):
    """Convert neutrino masses from eV to the reduced energy parameter m_bar.

    m_bar = m c²/k_b T

    While the rest of the code is generic, this function specifically
    assume 2 massless species and 1 massive specie bearing the sum of
    masses.

    Parameters:
    -----------
    m_nu: float
        sum of neutrino masses in eV
    T_nu: float
        neutrino temperature today in K.

    Returns:
    --------
    jnp.array
        Array of reduced mass parameters for neutrinos.

    """
    return (
        jnp.array([m_nu, 0.0, 0.0])
        * Constants.e
        / (Constants.k * T_nu)
    )


def analytical_small_mass_expansion(m_bar):
    """
    Analytical expansion for small mass parameter.

    Parameters:
    -----------
    m_bar : float or array
        Reduced mass parameter.

    Returns:
    --------
    float or array
        Approximation of the integral for small m_bar.
    """
    return 7 * jnp.pi**4 / 120 * (1 + 5 / (7 * jnp.pi**2) * m_bar**2)


def analytical_large_mass_expansion(m_bar):
    """
    Analytical expansion for large mass parameter.

    Parameters:
    -----------
    m_bar : float or array
        Reduced mass parameter.

    Returns:
    --------
    float or array
        Approximation of the integral for large m_bar.
    """
    return 3.0 / 2 * Constants.zeta3 * m_bar + 3 / (4 * m_bar) * 15 * Constants.zeta5


# Tabulated functions
n_chebyshev = 35
chebyshev_nodes_mass = chebyshev_nodes(n_chebyshev, -2, 3)
_interpolant = newton_interp(
    chebyshev_nodes_mass,
    compute_fermion_distribution_integral(10**chebyshev_nodes_mass),
)
interpolant = lambda x: _interpolant(jnp.log10(x))


@safe_vmap(in_axes=(0,))
def compute_composite_integral(x):
    """
    Compute the integral using a composite approach of analytical expansions and interpolation.

    Parameters:
    -----------
    x : float or array
        Reduced mass parameter.

    Returns:
    --------
    float or array
        The composite integral result.
    """
    # Compute the index based on x
    mass_thresholds = jnp.array([0.01, 1000])
    index = jnp.digitize(x, mass_thresholds)

    # Define branches
    branches = [
        analytical_small_mass_expansion,
        interpolant,
        analytical_large_mass_expansion,
    ]

    # Use lax.switch to select the appropriate branch
    return jax.lax.switch(index, branches, x)


# from cosmologix import densities
# from cosmologix.tools import Constants, trapezoidal_rule_integration, safe_vmap
# from cosmologix.interpolation import (
#    chebyshev_nodes,
#    newton_interp,
#    newton_divided_differences,
# )
# import jax.numpy as jnp
# import jax
#
#
# def rho_gamma_0(params):
#    """Energy density of CMB photons today in kg/m^3"""
#    return 4 * Constants.sigma * params["Tcmb"] ** 4 / Constants.c**3
#
#
# def T_nu(params):
#    """Neutrinos distribution temperature today
#
#    See 2005NuPhB.729..221M
#    """
#    params["T_nu"] = (
#        (4 / 11) ** (1.0 / 3) * (params["Neff"] / 3) ** (1.0 / 4) * params["Tcmb"]
#    )
#    return params
#
#
# def rho_nu_i_relativistic(params):
#    return 7.0 / 8 * params["Neff"] / 3 * (4 / 11) ** (4.0 / 3) * rho_gamma_0(params)
#
#
# @safe_vmap()
# def rho_nu(params, z):
#    return (
#        rho_nu_i_relativistic(params)
#        * 120
#        / (7 * jnp.pi**4)
#        * composite_I(params["m_nu_bar"] / (1 + z))
#    )
#
#
# @safe_vmap(in_axes=(0,))
# def I_m(m_bar):
#    r"""Helping function to compute the integral of the energy
#    distribution of massive fermions.
#
#    This function of the reduced mass \bar m evaluates:
#    \int_0^\inf x^3 \sqrt(1 + (m_bar/x)^2)/(e^x + 1) dx
#
#    Notes:
#    ------
#
#    This function admits expansions for the non-relativistics and
#    relativistic case. It is also very smooth in between so that it
#    can be sped-up by orders of magnitudes by combining the expansions
#    with a precomputed polynomial interpolant. This is what is done in
#    composite_I.
#    """
#
#    def integrand(x):
#        return x**3 * jnp.sqrt(1 + (m_bar / x) ** 2) / (1 + jnp.exp(x))
#
#    return trapezoidal_rule_integration(integrand, 1e-3, 31, 10000)
#
#
# def m_bar(params):
#    """Convert neutrinos masses from eV to reduced energy parameter m_bar
#
#    m_bar = m c²/k_b T
#    """
#    return (
#        jnp.array([params["m_nu"], 0.0, 0.0])
#        * Constants.e
#        / (Constants.k * params["T_nu"])
#    )
#
#
# def analytical_m_small(m_bar):
#    return 7 * jnp.pi**4 / 120 * (1 + 5 / (7 * jnp.pi**2) * m_bar**2)
#
#
# def analytical_m_large(m_bar):
#    return 3.0 / 2 * Constants.zeta3 * m_bar + 3 / (4 * m_bar) * 15 * Constants.zeta5
#
#
## Tabulated functions
# n_cheb = 35
# bar_nodes = chebyshev_nodes(n_cheb, -2, 3)
# _interp2 = newton_interp(bar_nodes, I_m(10**bar_nodes))
# interp2 = lambda x: _interp2(jnp.log10(x))
#
#
#
## Define the composite function using lax.switch
# @safe_vmap(in_axes=(0,))
# def composite_I(x):
#    # Compute the index based on x
#    index = jnp.digitize(x, jnp.array([0.01, 1000]))
#
#    # Define branches
#    branches = [analytical_m_small, interp2, analytical_m_large]
#
#    # Use lax.switch to select the appropriate branch
#    return jax.lax.switch(index, branches, x)
#
