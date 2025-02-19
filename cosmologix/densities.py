import jax.numpy as jnp
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
        - 3 * params["wa"] / (1 + z)
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


def params_to_density_params(params):
    """
    Convert cosmological parameters to density parameters.

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
    """
    params["Omega_b"] = params["Omega_b_h2"] / (params["H0"] / 100) ** 2
    params["Omega_gamma"] = neutrinos.compute_cmb_photon_density(params) / rhoc(
        params["H0"]
    )
    params = neutrinos.compute_neutrino_temperature(params)
    params["m_nu_bar"] = neutrinos.convert_mass_to_reduced_parameter(params)
    rho_nu = neutrinos.compute_neutrino_density(params, jnp.array([0])) / rhoc(params["H0"])
    massless = params["m_nu_bar"] == 0
    params["Omega_nu_massless"] = rho_nu[:, massless].sum().item()
    params["Omega_nu_massive"] = rho_nu[:, ~massless].sum().item()
    params["Omega_c"] = params["Omega_m"] - params["Omega_b"] - params["Omega_nu_massive"]
    params["Omega_x"] = (
        1
        - params["Omega_k"]
        - params["Omega_m"]
        - params["Omega_gamma"]
        - params["Omega_nu_massless"]
    )
    return params


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
    params = params_to_density_params(params)
    return (
        Omega_c(params, z)
        + Omega_b(params, z)
        + Omega_gamma(params, z)
        + Omega_nu(params, z)
        + Omega_de(params, z)
        + Omega_k(params, z)
    )

