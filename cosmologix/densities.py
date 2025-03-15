import jax.numpy as jnp
import jax
from cosmologix.tools import safe_vmap, Constants
from cosmologix import neutrinos


def rhoc(H: float) -> float:
    """
    Calculate the critical density in kg/m^3.

    Parameters:
    -----------
    H : float
        Hubble parameter in km/s/Mpc.

    Returns:
    --------
    float
        Critical density in kg/m^3.
    """
    return 3 * (H * 1e-3 / Constants.pc) ** 2 / (8 * jnp.pi * Constants.G)


def Omega_c(params, z):
    """
    Calculate the cold dark matter density parameter at redshift z.

    Parameters:
    -----------
    params : dict
        A dictionary containing cosmological parameters, including 'Omega_c'.
    z : float or array
        Redshift.

    Returns:
    --------
    float or array
        Omega_c at given redshift z.
    """
    return params["Omega_c"] * (1 + z) ** 3


def Omega_b(params, z):
    """
    Calculate the baryon density parameter at redshift z.

    Parameters:
    -----------
    params : dict
        A dictionary containing cosmological parameters, including 'Omega_b'.
    z : float or array
        Redshift.

    Returns:
    --------
    float or array
        Omega_b at given redshift z.
    """
    return params["Omega_b"] * (1 + z) ** 3


def Omega_gamma(params, z):
    """
    Calculate the photon density parameter at redshift z.

    Parameters:
    -----------
    params : dict
        A dictionary containing cosmological parameters, including 'Omega_gamma'.
    z : float or array
        Redshift.

    Returns:
    --------
    float or array
        Omega_gamma at given redshift z.
    """
    return params["Omega_gamma"] * (1 + z) ** 4


def Omega_de(params, z):
    """
    Calculate the dark energy density parameter at redshift z for a CPL parameterization.

    Parameters:
    -----------
    params : dict
        A dictionary containing cosmological parameters, including 'Omega_x', 'w', and 'wa'.
    z : float or array
        Redshift.

    Returns:
    --------
    float or array
        Omega_de at given redshift z.
    """
    return params["Omega_x"] * jnp.exp(
        3 * (1 + params["w"] + params["wa"]) * jnp.log(1 + z)
        - 3 * params["wa"] * z / (1 + z)
    )


def Omega_k(params, z):
    """
    Calculate the curvature density parameter at redshift z.

    Parameters:
    -----------
    params : dict
        A dictionary containing cosmological parameters, including 'Omega_k'.
    z : float or array
        Redshift.

    Returns:
    --------
    float or array
        Omega_k at given redshift z.
    """
    return params["Omega_k"] * (1 + z) ** 2


def Omega_nu_massless(params, z):
    """
    Calculate the density parameter for massless neutrinos at redshift z.

    Parameters:
    -----------
    params : dict
        A dictionary containing cosmological parameters, including 'Omega_nu'.
    z : float or array
        Redshift.

    Returns:
    --------
    float or array
        Omega_nu for massless neutrinos at given redshift z.
    """
    return params["Omega_nu"] * (1 + z) ** 4


def Omega_nu(params, z):
    """
    Calculate the density parameter for massive neutrinos at redshift z.

    Parameters:
    -----------
    params : dict
        A dictionary containing cosmological parameters.
    z : float or array
        Redshift.

    Returns:
    --------
    float or array
        Omega_nu for massive neutrinos at given redshift z.
    """
    return (
        neutrinos.compute_neutrino_density(params, z).sum(axis=1).squeeze()
        * (1 + z) ** 4
        / rhoc(params["H0"])
    )


def process_params(params):
    """Process the set of primary parameters to add a set of derived quantities.

    This function updates the input dictionary with calculated density parameters
    based on the given cosmological parameters.

    Parameters:
    -----------
    params : dict
        A dictionary of cosmological parameters.

    Returns:
    --------
    dict
        Updated dictionary with density parameters.

    Note:
    -----

    The Planck and CCL convention to count Omega_nu_massive as a
    contribution to Omega_m is very inconvenient in the context of our
    jax computations because it causes branching issues around m_nu =
    0. Therefore we follow the convention that Omega_m = Omega_c +
    Omega_b. To count Omega_nu_massive and Omega_nu_massless, use the
    function derived_parameters.
    """
    derived_params = params.copy()
    derived_params["Omega_b"] = params["Omega_b_h2"] / (params["H0"] / 100) ** 2
    derived_params["Omega_gamma"] = neutrinos.compute_cmb_photon_density(
        params["Tcmb"]
    ) / rhoc(params["H0"])
    derived_params["T_nu"] = neutrinos.compute_neutrino_temperature(
        params["Tcmb"], params["Neff"]
    )
    derived_params["m_nu_bar"] = neutrinos.convert_mass_to_reduced_parameter(
        params["m_nu"], derived_params["T_nu"]
    )
    derived_params["Omega_c"] = params["Omega_m"] - derived_params["Omega_b"]
    derived_params["Omega_nu"] = Omega_nu(derived_params, jnp.array([0]))[0]
    derived_params["Omega_x"] = (
        1
        - derived_params["Omega_k"]
        - derived_params["Omega_m"]
        - derived_params["Omega_gamma"]
        - derived_params["Omega_nu"]
    )

    return derived_params


def derived_parameters(params):
    """Further decomposition of Omega_nu between massless and massive_neutrinos"""
    params = process_params(params)
    rho_nu = neutrinos.compute_neutrino_density(params, jnp.array([0])) / rhoc(
        params["H0"]
    )
    massless = params["m_nu_bar"] == 0
    params["Omega_nu_massless"] = rho_nu[:, massless].sum().item()
    params["Omega_nu_massive"] = rho_nu[:, ~massless].sum().item()
    return params


@jax.jit
def Omega(params, z):
    """
    Compute the total density parameter Omega for all components at given redshift(s).

    Parameters:
    -----------
    params : dict
        A dictionary containing all necessary cosmological parameters.
    z : float or array
        Redshift or array of redshifts.

    Returns:
    --------
    float or array
        Total Omega at the given redshift(s).
    """
    params = process_params(params)
    return (
        Omega_c(params, z)
        + Omega_b(params, z)
        + Omega_gamma(params, z)
        + Omega_nu(params, z)
        + Omega_de(params, z)
        + Omega_k(params, z)
    )
