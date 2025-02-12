import matplotlib.pyplot as plt
from cosmologix import mu, Planck18
import pyccl as ccl
from cosmologix.distances import Omega_c, Omega_de
import jax.numpy as jnp
import numpy as np
import jax
import camb
from cosmologix.tools import Constants

#
# Convenience functions to facilitate comparisons with CAMB and CCL
#


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
        "n_s": 0.9652,
        "sigma8": 0.8101,
        "transfer_function": "bbks",
    }


def params_to_CAMB(params):
    omegac = float(Omega_c(params))
    h = params["H0"] / 100
    pars = camb.set_params(
        H0=params["H0"],
        ombh2=params["Omega_b_h2"],
        omch2=omegac * h**2,
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


def camb_densities(params, z):
    pars = params_to_CAMB(params)
    results = camb.get_results(pars)
    r = results.get_background_densities(1 / (1 + z))
    # for s in r:
    #    r[s] = r[s] * (1+z)**4 / (8 * jnp.pi * Constants.G)
    translate = {
        "curvature": r["K"] / r["tot"],
        "matter": (r["cdm"] + r["baryon"] + r["nu"]) / r["tot"],
        "dark_energy": r["de"] / r["tot"],
        "radiation": r["photon"] / r["tot"],
        "neutrinos_rel": (r["neutrino"]) / r["tot"],
        "neutrinos_massive": r["nu"] / r["tot"],
    }

    return translate


def ccl_densities(params, z):
    cclcosmo = ccl.Cosmology(**params_to_ccl(params))
    species = [
        "matter",
        "dark_energy",
        "radiation",
        "curvature",
        "neutrinos_rel",
        "neutrinos_massive",
    ]
    crit = ccl.background.rho_x(cclcosmo, 1 / (1 + z), "critical")
    result = dict(
        [
            (specie, ccl.background.rho_x(cclcosmo, 1 / (1 + z), specie) / crit)
            for specie in species
        ]
    )
    return result


def distance_accuracy(params=Planck18.copy(), title="distance_accuracy"):
    comparisons = {
        "ccl": mu_ccl,
        "camb": mu_camb,
        "cosmologix coarse (1000)": lambda params, z: mu(params, z, 1000),
    }
    z = jnp.linspace(0.01, 1000, 3000)
    fig = plt.figure(title)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    for label, mu_alt in comparisons.items():
        ax1.plot(z, mu_alt(params, z), label=label)
        ax2.plot(z, mu_alt(params, z) - mu(params, z, 10000), label=label)
    ax1.plot(z, mu(params, z), "k-")
    ax2.axhline(0, color="k")
    ax1.set_xscale("log")
    ax1.legend(frameon=False, loc="best")
    ax2.set_xlabel(r"$z$")
    ax2.set_ylabel(r"$\mu - \mu_{baseline}$")
    ax1.set_ylabel(r"$\mu$")


def densities(params=Planck18.copy(), title="densities"):
    fig = plt.figure(title)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    z = np.logspace(np.log10(0.01), np.log10(1000), 3000)
    comp = {
        "camb": camb_densities,
        #'ccl': ccl_densities,
    }
    ref = ccl_densities(params, z)
    for label, code in comp.items():
        dens = code(Planck18, z)
        for species in dens:
            ax1.plot(z, dens[species], label=species)
            ax2.plot(z, dens[species] / ref[species])
    ax1.legend(loc="best", frameon=False)


def lcdm_deviation(**keys):
    params = Planck18.copy()
    params.update(keys)
    return params


if __name__ == "__main__":
    plt.ion()
    distance_accuracy()
    distance_accuracy(
        lcdm_deviation(m_nu=0), title="distance_accuracy (massless neutrinos)"
    )
    # densities(lcdm_deviation(m_nu=0))

    plt.show()
