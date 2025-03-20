from cosmologix.acoustic_scale import rs, z_star, z_drag, theta_MC, dM, dsound_da_approx
from cosmologix import Planck18
from test_distances import params_to_CAMB, lcdm_deviation
import pyccl as ccl
import jax
import camb
import jax.numpy as jnp


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from cosmologix.distances import dV, dM, dH
    from cosmologix.acoustic_scale import z_drag, rs, rd_approx
    from cosmologix.likelihoods import DESI2024DR1Prior

    z = jnp.linspace(0.05, 2.5, 100)
    rd1 = rs(Planck18, z_drag(Planck18))
    rd2 = rd_approx(Planck18)

    desi = DESI2024DR1Prior()
    prediction = desi.model(Planck18)

    desi_DV = desi.distances[desi.dist_type_indices == 0]
    desi_DM = desi.distances[desi.dist_type_indices == 1]
    desi_DH = desi.distances[desi.dist_type_indices == 2]

    fig = plt.figure(figsize=(8, 8))
    plt.plot(z, dV(Planck18, z) / (rd1 * z ** (2 / 3)))
    plt.plot(z, dV(Planck18, z) / (rd2 * z ** (2 / 3)))
    desi_DV_err = jnp.sqrt(
        desi.cov[desi.dist_type_indices == 0, desi.dist_type_indices == 0]
    )
    desi_z = desi.redshifts[desi.dist_type_indices == 0]
    plt.errorbar(
        desi_z,
        desi_DV / desi_z ** (2 / 3),
        yerr=desi_DV_err / desi_z ** (2 / 3),
        fmt="o",
    )
    desi_z = jnp.unique(desi.redshifts[desi.dist_type_indices != 0])
    desi_DV = (desi_z * desi_DM**2 * desi_DH) ** (1 / 3)
    desi_DV_err = desi_DV * jnp.sqrt(
        4
        / 9
        * desi.cov[desi.dist_type_indices == 1, desi.dist_type_indices == 1]
        / desi_DM**2
        + 1
        / 9
        * desi.cov[desi.dist_type_indices == 2, desi.dist_type_indices == 2]
        / desi_DH**2
    )
    plt.errorbar(
        desi_z,
        desi_DV / desi_z ** (2 / 3),
        yerr=desi_DV_err / desi_z ** (2 / 3),
        fmt="o",
    )
    plt.show()

    fig2 = plt.figure(figsize=(8, 8))
    plt.plot(z, dM(Planck18, z) / (z * dH(Planck18, z)))
    desi_z = jnp.unique(desi.redshifts[desi.dist_type_indices != 0])
    desi_DM_over_DH = desi_DM / desi_DH / desi_z
    desi_DM_over_DH_err = desi_DM_over_DH * jnp.sqrt(
        desi.cov[desi.dist_type_indices == 1, desi.dist_type_indices == 1] / desi_DM**2
        + desi.cov[desi.dist_type_indices == 2, desi.dist_type_indices == 2]
        / desi_DH**2
    )
    plt.errorbar(desi_z, desi_DM_over_DH, yerr=desi_DM_over_DH_err, fmt="o")
    plt.show()
