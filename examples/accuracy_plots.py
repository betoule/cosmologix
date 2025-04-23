import matplotlib.pyplot as plt
from cosmologix import mu, Planck18
import pyccl as ccl
from cosmologix import neutrinos, densities
import jax.numpy as jnp
import numpy as np
import jax
import camb
from cosmologix.tools import Constants
from astropy import cosmology
import jax_cosmo as jc

try:
    import cosmoprimo
except ImportError:
    print(
        "Running the full comparison requires manual installation of cosmoprimo, for example via python -m pip install git+https://github.com/cosmodesi/cosmoprimo"
    )


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


def params_to_astropy(params):
    params = densities.derived_parameters(params)
    h = params["H0"] / 100.0
    # Omega_b = params['Omega_b_h2'] / h ** 2
    # Omega_nu_mass = float(Omega_n_mass(params, 1.)[0])
    return cosmology.w0waCDM(
        H0=params["H0"],
        Om0=params["Omega_bc"],
        Ob0=params["Omega_b"],
        Ode0=params["Omega_x"],
        m_nu=[params["m_nu"], 0, 0],
        Tcmb0=params["Tcmb"],
        Neff=params["Neff"],
        w0=params["w"],
        wa=params["wa"],
    )


def mu_camb(params, z):
    pars = params_to_CAMB(params)
    results = camb.get_background(pars)  # camb.get_results(pars)
    return 5 * jnp.log10(results.luminosity_distance(z)) + 25


def mu_ccl(params, z):
    cclcosmo = ccl.Cosmology(**params_to_ccl(params))
    return ccl.distance_modulus(cclcosmo, 1 / (1 + z))


def mu_astropy(params, z):
    astropycosmo = params_to_astropy(params)
    return astropycosmo.distmod(np.asarray(z)).value


@jax.jit
def mu_jaxcosmo(params, z):
    h = params["H0"] / 100
    omega_b = params["Omega_b_h2"] / h**2
    a = 1 / (1 + z)
    jaxcosmo = jc.Cosmology(
        Omega_c=params["Omega_bc"] - omega_b,
        Omega_b=omega_b,
        h=h,
        Omega_k=params["Omega_k"],
        n_s=0.96,
        sigma8=0.8,
        w0=params["w"],
        wa=params["wa"],
    )
    return 5 * jnp.log10(
        jc.background.radial_comoving_distance(jaxcosmo, a, steps=8 * 1024)
        * 1e5
        * (1 + z)
        / h
    )


# @jax.jit
def mu_cosmoprimo(params, z):
    h = params["H0"] / 100
    omega_b = params["Omega_b_h2"] / h**2
    c = cosmoprimo.Cosmology(
        engine="eisenstein_hu",
        h=h,
        Omega_b=omega_b,
        Omega_cdm=params["Omega_bc"] - omega_b,
        Omega_k=params["Omega_k"],
        Tcmb=params["Tcmb"],
        w0_fld=params["w"],
        wa_fld=params["wa"],
        m_ncdm=params["m_nu"],
    )

    b = c.get_background()
    return 5 * jnp.log10(b.luminosity_distance(z) / h) + 25


def camb_densities(params, z):
    pars = params_to_CAMB(params)
    results = camb.get_results(pars)
    r = results.get_background_densities(1 / (1 + z))
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
    crit = ccl.background.rho_x(cclcosmo, 1.0, "critical")
    result = dict(
        [
            (specie, ccl.background.rho_x(cclcosmo, 1 / (1 + z), specie) / crit)
            for specie in species
        ]
    )
    return result


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


def distance_accuracy(params=Planck18.copy(), title="distance_accuracy"):
    comparisons = {
        "ccl": mu_ccl,
        "camb": mu_camb,
        "astropy": mu_astropy,
        "cosmologix coarse (1000)": lambda params, z: mu(params, z, 1000),
        "cosmoprimo": mu_cosmoprimo,
        # "jax_cosmo": mu_jaxcosmo,
    }
    z = jnp.logspace(-2, 3, 3000)
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
    ax2.set_ylim([-1e-4, 1e-4])
    ax1.set_ylabel(r"$\mu$")


def plot_densities(params=Planck18.copy(), title="densities"):
    fig = plt.figure(title)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    z = np.logspace(np.log10(0.01), np.log10(1000), 3000)
    comp = {
        # "camb": camb_densities,
        "cosmologix": cosmologix_densities,
        #'ccl': ccl_densities,
    }
    ref = ccl_densities(params, z)
    for label, code in comp.items():
        dens = code(params, z)
        for species in dens:
            # if species.startswith('neutrinos'):
            #    continue
            ax1.plot(z, dens[species], label=species)
            ax2.plot(z, dens[species] / ref[species])
    ax1.legend(loc="best", frameon=False)
    return ref, dens


def lcdm_deviation(**keys):
    params = Planck18.copy()
    params.update(keys)
    return params


if __name__ == "__main__":
    plt.ion()
    plt.rc("text", usetex=True)
    plt.rc("axes.spines", top=False, right=False, bottom=True, left=True)

    distance_accuracy()
    plt.tight_layout()
    plt.savefig("doc/mu_accuracy.svg")
    distance_accuracy(
        lcdm_deviation(m_nu=0.0), title="distance_accuracy (massless neutrinos)"
    )
    # ref, dens = densities(lcdm_deviation(m_nu=0., wa=0.))
    ref, dens = plot_densities(lcdm_deviation(wa=0.0))
    plt.show()
