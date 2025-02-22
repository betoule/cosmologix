from cosmologix import likelihoods, Planck18
from cosmologix.contours import (
    frequentist_contour_2D_sparse,
    color_theme,
    plot_contours,
)
import matplotlib.pyplot as plt

plt.ion()


# Define the explored space (a regular grid in the plane Omega_m/w
# with 60 points between the boundaries)
n = 60
param_space = {"Omega_m": [0.18, 0.48, n], "w": [-0.6, -1.5, n]}

# Define the parameter kept fixed during the exploration. Here we are
# exploring a one parameter (free but constant w) expansion of the
# baseline flat-ΛCDM model
fixed = {"Omega_k": 0.0, "m_nu": 0.06, "Neff": 3.046, "Tcmb": 2.7255, "wa": 0.0}

# Compute BAO constraints keeping Omega_b_h2 fixed to the best-fit
# Planck value
grid_bao = frequentist_contour_2D_sparse(
    [likelihoods.DESI2024Prior(uncalibrated=True)],
    grid=param_space,
    fixed=dict(fixed, Omega_b_h2=Planck18["Omega_b_h2"], H0=Planck18["H0"]),
)

# Compute SN constraints. H0 is not constrained by the data, because
# it is totally degenerate with the (varied) absolute luminosity of
# supernovae. Baryon and cold dark matter densities have exactly the
# same impact on the expansion rate, so that only the sum of the two
# is constrained by the luminosity distance measurement.
grid_sn = frequentist_contour_2D_sparse(
    [likelihoods.DES5yr()],
    grid=param_space,
    fixed=dict(fixed, H0=Planck18["H0"], Omega_b_h2=Planck18["Omega_b_h2"]),
)

grid_jla = frequentist_contour_2D_sparse(
    [likelihoods.JLA()],
    grid=param_space,
    fixed=dict(fixed, H0=Planck18["H0"], Omega_b_h2=Planck18["Omega_b_h2"]),
)

# Compute CMB constraints. The "geometric" summary used in this code
# only constrain the angular distance to the last diffusion surface.,
grid_cmb = frequentist_contour_2D_sparse(
    [likelihoods.Planck2018Prior()],
    grid=param_space,
    fixed=fixed,
)

cmb_bao = frequentist_contour_2D_sparse(
    [likelihoods.Planck2018Prior(), likelihoods.DESI2024Prior()],
    grid=param_space,
    fixed=fixed,
)

cmb_sn = frequentist_contour_2D_sparse(
    [likelihoods.Planck2018Prior(), likelihoods.JLA()],
    grid=param_space,
    fixed=fixed,
)
# Compute constraints from a combination of CMB and SN data.
priors = [likelihoods.Planck2018Prior(), likelihoods.DES5yr()]
grid = frequentist_contour_2D_sparse(priors, grid=param_space, fixed=fixed)

# Plot the result
plt.rc("text", usetex=True)
plt.rc("axes.spines", top=False, right=False, bottom=False, left=False)
plot_contours(grid_bao, base_color=color_theme[0], filled=True, label="DESI")
plot_contours(grid_cmb, base_color=color_theme[1], filled=True, label="Planck")
plot_contours(grid_jla, base_color=color_theme[2], filled=True, label="JLA")
plot_contours(
    cmb_bao, base_color="black", filled=False, label="Planck+DES5", bestfit=True
)
plot_contours(cmb_sn, base_color="red", filled=False, label="Planck+JLA", bestfit=True)
plt.axhline(-1, color="k", ls=":", lw=0.5)
plt.text(0.48, -1, r"$\Lambda$-CDM", ha="right", va="bottom")
plt.legend(loc="lower right", frameon=False)

plt.show()
