from cosmologix import mu, Planck18, densities
import pyccl as ccl
import jax.numpy as jnp
import jax
import camb

# Set the default precision to float64 for all operations
jax.config.update("jax_enable_x64", True)


def lcdm_deviation(**keys):
    params = Planck18.copy()
    params.update(keys)
    return params


massless = lcdm_deviation(m_nu=0)
opened = lcdm_deviation(Omega_k=0.01)
closed = lcdm_deviation(Omega_k=-0.01)


#
# Convenience functions to facilitate comparisons with CAMB and CCL
#
def params_to_ccl(params):
    params = densities.derived_parameters(params)
    return {
        "Omega_c": params["Omega_c"],
        "Omega_b": params["Omega_b"],
        "Omega_k": params["Omega_k"],
        "h": params["H0"] / 100,
        "Neff": params["Neff"],
        "m_nu": [params["m_nu"], 0, 0],
        "T_CMB": params["Tcmb"],
        "T_ncdm": 0.7137658555036082,
        "n_s": 0.9652,
        "sigma8": 0.8101,
        "transfer_function": "bbks",
    }


def params_to_CAMB(params):
    params = densities.derived_parameters(params)
    h = params["H0"] / 100
    pars = camb.set_params(
        H0=params["H0"],
        ombh2=params["Omega_b_h2"],
        omch2=params["Omega_c"] * h**2,
        mnu=params["m_nu"],
        omk=params["Omega_k"],
        tau=0.0540,
        As=jnp.exp(3.043) / 10**10,
        ns=0.9652,
        halofit_version="mead",
        lmax=3000,
    )
    return pars


def mu_camb(params, z):
    pars = params_to_CAMB(params)
    results = camb.get_results(pars)
    return 5 * jnp.log10(results.luminosity_distance(z)) + 25


def mu_ccl(params, z):
    cclcosmo = ccl.Cosmology(**params_to_ccl(params))
    return ccl.distance_modulus(cclcosmo, 1 / (1 + z))


def test_distance_modulus():
    z = jnp.linspace(0.01, 1, 3000)
    for params in [Planck18, massless, opened, closed]:
        for mu_check in [mu_ccl, mu_camb]:
            delta_mu = mu(params, z) - mu_check(params, z)
            assert (
                jnp.abs(delta_mu) < 1e-3
            ).all(), f"Distances differs for cosmology {params}, {mu_check}"


if __name__ == "__main__":
    test_distance_modulus()
