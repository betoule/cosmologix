from test_distances import params_to_ccl
from cosmologix.tools import speed_measurement
from cosmologix import densities, Planck18, mu
import jax.numpy as jnp


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


if __name__ == "__main__":
    params = densities.process_params(Planck18)
    z = jnp.logspace(-2, 3, 1000)
    results = {}
    func_list = [
        "Omega_c",
        "Omega_b",
        "Omega_gamma",
        "Omega_nu",
        "Omega_de",
        "Omega_k",
        "Omega",
    ]
    funcdict = dict([(func, getattr(densities, func)) for func in func_list])
    funcdict["mu"] = mu
    for func, f in funcdict.items():

        results[func] = speed_measurement(f, params, z)
    print(results)

    import matplotlib.pyplot as plt

    labels = list(results.keys())
    r = jnp.arange(len(labels))
    bar_width = 0.25
    fig = plt.figure()
    ax = fig.subplots()
    for i, (label, num) in enumerate(
        [
            ("first call", 0),
            ("subsequent", 1),
            ("first call (jitted)", 3),
            ("subsequent (jitted)", 4),
        ]
    ):

        values = [results[k][num] for k in labels]
        # plt.barh(labels, values)
        ax.bar(r + i * bar_width, values, width=bar_width, label=label, alpha=0.8)
    ax.set_xticks(r + bar_width * (4 - 1) / 2)
    ax.set_xticklabels(labels)
