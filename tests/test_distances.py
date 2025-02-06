from cosmologix import mu, Planck18
import pyccl as ccl
from cosmologix.distances import Omega_c, Omega_de
import jax.numpy as jnp
import jax

# Set the default precision to float64 for all operations
jax.config.update("jax_enable_x64", True)


def lcdm_deviation(**keys):
    params = Planck18.copy()
    params.update(keys)
    return params


massless = lcdm_deviation(m_nu=0)
opened = lcdm_deviation(Omega_k=0.01)
closed = lcdm_deviation(Omega_k=-0.01)


def params_to_ccl(params):
    h = params["H0"] / 100
    return {
        "Omega_c": float(Omega_c(params)),
        "Omega_b": params["Omega_b_h2"] / h**2,
        "Omega_k": params["Omega_k"],
        "h": h,
        "Neff": params["Neff"],
        "m_nu": [params["m_nu"], 0, 0],
        "T_CMB": params["Tcmb"],
        "T_ncdm": 0.7137658555036082,
        "n_s": 0.96,
        "sigma8": 0.8,
        "transfer_function": "bbks",
    }


def mu_ccl(params, z):
    cclcosmo = ccl.Cosmology(**params_to_ccl(params))
    return ccl.distance_modulus(cclcosmo, 1 / (1 + z))


def test_distance_modulus():
    z = jnp.linspace(0.01, 1, 3000)
    for params in [Planck18, massless, opened, closed]:
        delta_mu = mu(params, z) - mu_ccl(params, z)
        assert (
            jnp.abs(delta_mu) < 1e-3
        ).all(), f"Distances differs for cosmology {params}"
