from cosmologix import (
    Planck18,
    likelihoods,
    lcdm_deviation,
    fit,
    distances,
    acoustic_scale,
)
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


def rd(params):
    return acoustic_scale.rs(params, acoustic_scale.z_drag(params))


# Uncalibrated DESI prior
desiu = likelihoods.DESI2024Prior(True)

# Prep for flat ΛCDM fit
fixed = lcdm_deviation()
fixed.pop("Omega_m")

results = fit([desiu], fixed=fixed, verbose=True)

# Reconstruct the full vector of parameters
aparams = lcdm_deviation(**results["bestfit"])

# Extract the Omega_m and error
Om = results["bestfit"]["Omega_m"]
eOm = jnp.sqrt(jnp.diag(results["FIM"]))[0]

# Compute residuals
res = desiu.residuals(aparams)
error = jnp.sqrt(jnp.diag(desiu.cov))


z = jnp.linspace(0.1, 2.5)

# Plot on the same plot the 3 kinds of BAO measurements with arbitrary
# scaling in redshift to avoid blowing up the scale
plt.ion()
plt.rc("text", usetex=True)
fig = plt.figure("DESI BAO measurements")
ax1, ax2 = fig.subplots(2, 1)
for dist_type, distfunc, zscale, label in [
    ("DV_over_rd", distances.dV, (2.0 / 3), r"$\frac{D_V}{r_d z^{2/3}}$"),
    ("DH_over_rd", distances.dH, (-2 / 3), r"$\frac{D_H}{r_d z^{-2/3}}$"),
    ("DM_over_rd", distances.dM, (2 / 3), r"$\frac{D_M}{r_d z^{2/3}}$"),
]:
    goods = np.array(desiu.dist_type_labels) == dist_type
    zdata = desiu.redshifts[goods]
    ydata = desiu.distances[goods]
    ey = error[goods]
    l = ax1.errorbar(
        zdata, ydata / zdata ** (zscale), ey / zdata ** (zscale), marker="o", ls="None"
    )
    ax1.plot(
        z,
        distfunc(aparams, z) / (aparams["rd"] * z ** (zscale)),
        color=l[0].get_color(),
        label=label,
    )
    ax1.plot(
        z,
        distfunc(Planck18, z) / (rd(Planck18) * z ** (zscale)),
        color=l[0].get_color(),
        ls="--",
    )
    ax2.errorbar(
        desiu.redshifts[goods], res[goods], error[goods], ls="None", marker="s"
    )
ax2.set_xlabel(r"$z$")
ax2.set_ylabel("residuals")
ax2.axhline(0, color="k", label=rf"$\Omega_m = {Om:.3f} ± {eOm:.3f}$")
ax1.legend(loc="lower right", frameon=False)
ax2.legend(loc="lower right", frameon=False)
