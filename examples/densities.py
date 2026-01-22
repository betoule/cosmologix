import matplotlib.pyplot as plt
import pyccl as ccl
from cosmologix import neutrinos, densities
from cosmologix.distances import mu
from cosmologix.parameters import get_cosmo_params
import jax.numpy as jnp
import numpy as np
import jax
from cosmologix.tools import Constants
from astropy import cosmology


def cosmologix_densities(params, z):
    params = densities.derived_parameters(params)
    rho_nu = (
        neutrinos.compute_neutrino_density(params, z)
        * (1 + z[:, None]) ** 4
        / densities.rhoc(params["H0"])
    )
    massless = params["m_nu_bar"] == 0
    return {
        "matter": densities.Omega_c(params, z)
        + densities.Omega_b(params, z)
        + rho_nu[:, ~massless].sum(axis=1),
        "dark_energy": densities.Omega_de(params, z),
        "radiation": densities.Omega_gamma(params, z),
        "curvature": densities.Omega_k(params, z),
        "neutrinos_rel": rho_nu[:, massless].sum(
            axis=1
        ),  # cosmologix.densities.Omega_nu_massless(params, z),
        "neutrinos_massive": rho_nu[:, ~massless].sum(axis=1),
    }


def plot_densities(params, title="densities"):
    fig = plt.figure(title)
    ax1 = fig.subplots(1, 1, sharex=True)
    z = np.logspace(np.log10(0.01), np.log10(1000), 3000)
    dens = cosmologix_densities(params, z)
    for specie in dens:
        ax1.plot(z, dens[specie], label=f"{specie}")
    ax1.legend(loc="best", frameon=False)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$z$", fontsize=16)
    ax1.set_ylabel(r"$\Omega_i(z)$", fontsize=16)
    plt.show()
    return dens


if __name__ == "__main__":
    plt.rc("text", usetex=True)
    plt.rc("axes.spines", top=False, right=False, bottom=True, left=True)


    dens = plot_densities(get_cosmo_params('Planck18'))

