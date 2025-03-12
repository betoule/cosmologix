import camb
from test_distances import params_to_CAMB, cosmologies
import numpy as np
import jax.numpy as jnp
from cosmologix import densities, distances
from astropy import cosmology


def params_to_astropy(params):
    params = densities.derived_parameters(params)
    h = params["H0"] / 100.0
    # Omega_b = params['Omega_b_h2'] / h ** 2
    # Omega_nu_mass = float(Omega_n_mass(params, 1.)[0])
    return cosmology.w0waCDM(
        H0=params["H0"],
        Om0=params["Omega_m"],
        Ob0=params["Omega_b"],
        Ode0=params["Omega_x"],
        m_nu=[params["m_nu"], 0, 0],
        Tcmb0=params["Tcmb"],
        Neff=params["Neff"],
        w0=params["w"],
        wa=params["wa"],
    )


def V_astropy(params, z):
    astropycosmo = params_to_astropy(params)
    return astropycosmo.comoving_volume(np.asarray(z)).value

def lookback_time_astropy(params, z):
    astropycosmo = params_to_astropy(params)
    return astropycosmo.lookback_time(np.asarray(z)).value

def dV_astropy(params, z):
    astropycosmo = params_to_astropy(params)
    return astropycosmo.differential_comoving_volume(np.asarray(z)).value


def test_volumes():
    z = jnp.linspace(0.01, 1, 3000)
    for label, params in cosmologies.items():
        dV_over_V = V_astropy(params, z) / distances.comoving_volume(params, z) - 1
        assert (
            jnp.abs(dV_over_V) < 1e-3
        ).all(), f"Volume differs for cosmology {label}, {dV_over_V}"

def test_lookback_time():
    z = jnp.linspace(0.01, 1, 3000)
    for label, params in cosmologies.items():
        dV_over_V = lookback_time_astropy(params, z) / distances.lookback_time(params, z) - 1
        assert (
            jnp.abs(dV_over_V) < 1e-3
        ).all(), f"Volume differs for cosmology {label}, {dV_over_V}"
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.ion()
    z = jnp.logspace(-2, 3, 1000)
    fig = plt.figure("comoving_volume")
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.loglog(z, V_astropy(cosmologies["flcdm"], z))
    ax1.loglog(z, distances.comoving_volume(cosmologies[f"flcdm"], z))
    for label, params in cosmologies.items():
        ax2.plot(
            z,
            V_astropy(params, z) / distances.comoving_volume(params, z) - 1,
            label=label,
        )
    ax2.legend(frameon=False)

    fig = plt.figure("differential_comoving_volume")
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.loglog(z, dV_astropy(cosmologies["flcdm"], z))
    ax1.loglog(z, distances.differential_comoving_volume(cosmologies[f"flcdm"], z))
    for label, params in cosmologies.items():
        ax2.plot(
            z,
            dV_astropy(params, z) / distances.differential_comoving_volume(params, z)
            - 1,
            label=label,
        )
    ax2.legend(frameon=False)

    plt.show()
