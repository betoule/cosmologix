from cosmologix import (
    likelihoods,
    parameters,
    fitter,
    distances,
    acoustic_scale,
)
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


# Uncalibrated DESI prior
desiu = likelihoods.DESIDR1(True)

# Prep for flat ΛCDM fit
fixed = parameters.get_cosmo_params()
fixed.pop("Omega_bc")

results = fitter.fit([desiu], fixed=fixed, verbose=True, initial_guess=parameters.get_cosmo_params())

# Reconstruct the full vector of parameters
aparams = parameters.get_cosmo_params(**results["bestfit"])

# Extract the Omega_bc and error
Om = results["bestfit"]["Omega_bc"]
eOm = jnp.sqrt(jnp.diag(results["inverse_FIM"]))[0]

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
    ydata = desiu.data[goods]
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
        distfunc(parameters.get_cosmo_params(), z) / (acoustic_scale.rd(parameters.get_cosmo_params()) * z ** (zscale)),
        color=l[0].get_color(),
        ls="--",
    )
    ax2.errorbar(
        desiu.redshifts[goods],
        res[goods] / desiu.model(aparams)[goods],
        error[goods] / desiu.model(aparams)[goods],
        ls="None",
        marker="s",
    )
ax2.set_xlabel(r"$z$")
ax2.set_ylabel("residuals/model")
ax2.axhline(0, color="k", label=rf"$\Omega_bc = {Om:.3f} ± {eOm:.3f}$")
ax1.legend(loc="lower right", frameon=False)
ax2.legend(loc="lower right", frameon=False)
