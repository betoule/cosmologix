import matplotlib.pyplot as plt
import numpy as np

from cosmologix import (
    acoustic_scale,
    distances,
    fitter,
    likelihoods,
    parameters,
    contours,
    display,
    densities,
)

import jax.numpy as jnp


###########################################
# Convenience functions to change variables
###########################################
def rdtohrd(rd, h=parameters.get_cosmo_params()["H0"] / 100):
    hrd = h * rd
    return hrd


def hrdtord(hrd, h=parameters.get_cosmo_params()["H0"] / 100):
    rd = hrd / h
    return rd


def omegabc_to_omegam(
    omegabc,
    omega_nu=densities.derived_parameters(parameters.get_cosmo_params())["Omega_nu"],
):
    return omegabc + omega_nu


###########################################

# Uncalibrated DESI prior
desiu = likelihoods.DESIDR2(True)

# Prep for flat ΛCDM fit
fixed = parameters.get_cosmo_params()
fixed.pop("Omega_bc")

results = fitter.fit(
    [desiu], fixed=fixed, verbose=True, initial_guess=parameters.get_cosmo_params()
)

# Reconstruct the full vector of parameters
aparams = results[
    "bestfit_full"
]  # parameters.get_cosmo_params(**results["bestfit_full"])

# Extract the omega_m, hrd and error
Om = omegabc_to_omegam(results["bestfit"]["Omega_bc"])
rdh = rdtohrd(results["bestfit_full"]["rd"], h=results["bestfit_full"]["H0"] / 100)
eOm = results["uncertainties"]["Omega_bc"]
erdh = rdtohrd(results["uncertainties"]["rd"], h=results["bestfit_full"]["H0"] / 100)

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
        distfunc(parameters.get_cosmo_params(), z)
        / (acoustic_scale.rd(parameters.get_cosmo_params()) * z ** (zscale)),
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
    ax2.plot(
        z,
        (
            distfunc(parameters.get_cosmo_params(), z)
            / (acoustic_scale.rd(parameters.get_cosmo_params()) * z ** (zscale))
        )
        / (distfunc(aparams, z) / (aparams["rd"] * z ** (zscale)))
        - 1,
        color=l[0].get_color(),
        ls="--",
    )

ax2.set_xlabel(r"$z$")
ax2.set_ylabel("residuals/best fit")
ax2.axhline(
    0,
    color="k",
    label=rf"$\Omega_m = {Om:.4f} \pm {eOm:.4f}$"
    "\n"
    rf"$r_d h = {rdh:.2f} \pm {erdh:.2f}$",
)
ax1.legend(loc="lower right", frameon=False)
ax2.legend(loc="lower right", frameon=False)


######################################################
# Let's split the DESIDR2 prior by sub-samples to
# reproduce Fig. 7 in https://arxiv.org/pdf/2503.14738
######################################################
def subprior(index):
    mask = np.array(index)
    return likelihoods.UncalibratedBAOLikelihood(
        desiu.redshifts[mask],
        desiu.data[mask],
        desiu.cov[np.ix_(mask, mask)],
        [desiu.dist_type_labels[m] for m in mask],
    )


subsamples = {
    "BGS": ([0], "green"),
    "LRG1": ([1, 2], "yellow"),
    "LRG2": ([3, 4], "orange"),
    "LRG3+ELG1": ([5, 6], "darkblue"),
    "ELG2": ([7, 8], "cyan"),
    "QSO": ([9, 10], "darkgreen"),
    r"Ly$\alpha$": ([11, 12], "purple"),
    "All": (list(range(13)), "black"),
}


prior = subprior([0])
bf = fitter.fit([prior], fixed={**fixed, "rd": 151.0})
assert np.isfinite(bf["loss"][-1])
baocontours = {}
for label, index in subsamples.items():
    prior = subprior(index[0])
    contour = contours.frequentist_contour_2d(
        [prior],
        grid={"rd": [hrdtord(85), hrdtord(115), 100], "Omega_bc": [0.15, 0.55, 100]},
        fixed=fixed,
    )
    baocontours[label] = contour


plt.figure("Fig 7.")
for label, contour in baocontours.items():
    color = subsamples[label][1]
    display.plot_contours(
        contour,
        label=label,
        color=color,
        filled=label=='All',
        transform={"rd": ("hrd", rdtohrd), "Omega_bc": ("Omega_m", omegabc_to_omegam)},
    )
plt.legend(loc="best", frameon=False)

plt.show(block=True)
