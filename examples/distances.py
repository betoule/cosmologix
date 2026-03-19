import matplotlib.pyplot as plt
import numpy as np

from cosmologix import densities, distances
from cosmologix.parameters import get_cosmo_params


def cosmologix_distances(params, z):
    params = densities.derived_parameters(params)
    return {
        "angular distance": distances.dA(params, z),
        "luminosity distance": distances.dL(params, z),
        "transverse comoving distance": distances.dM(params, z),
        "hubble distance": distances.dH(params, z),
        "volumic distance": distances.dV(params, z),
    }


def plot_distances(params, title="distances"):
    fig = plt.figure(title)
    ax1 = fig.subplots(1, 1, sharex=True)
    z = np.logspace(np.log10(0.01), np.log10(1000), 3000)
    dists = cosmologix_distances(params, z)
    for d in dists:
        ax1.plot(z, dists[d], label=f"{d}")
    ax1.legend(loc="best", frameon=False)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$z$", fontsize=16)
    ax1.set_ylabel(r"Distances [Mpc]", fontsize=16)
    plt.show()
    return dists


if __name__ == "__main__":
    plt.rc("text", usetex=True)
    plt.rc("axes.spines", top=False, right=False, bottom=True, left=True)

    dists = plot_distances(get_cosmo_params("Planck18"))
