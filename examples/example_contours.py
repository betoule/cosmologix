from cosmologix import likelihoods, parameters, contours, tools, display
import matplotlib.pyplot as plt

plt.ion()


# Define the explored space (a regular grid in the plane Omega_bc/w
# with 60 points between the boundaries)
n = 60
param_space = {"Omega_bc": [0.18, 0.48, n], "w": [-0.6, -1.5, n]}

# Define the parameter kept fixed during the exploration. Here we are
# exploring a one parameter (free but constant w) expansion of the
# baseline flat-ΛCDM model
fixed = {"Omega_k": 0.0, "m_nu": 0.06, "Neff": 3.046, "Tcmb": 2.7255, "wa": 0.0}

# Compute BAO constraints keeping Omega_b_h2 fixed to the best-fit
# Planck value
grid_bao = contours.frequentist_contour_2d_sparse(
    [likelihoods.DESIDR2(uncalibrated=True)],
    grid=param_space,
    fixed=dict(
        fixed,
        Omega_b_h2=parameters.Planck18["Omega_b_h2"],
        H0=parameters.Planck18["H0"],
    ),
)
tools.save(grid_bao, "contour_desi.pkl")
# Compute SN constraints. H0 is not constrained by the data, because
# it is totally degenerate with the (varied) absolute luminosity of
# supernovae. Baryon and cold dark matter densities have exactly the
# same impact on the expansion rate, so that only the sum of the two
# is constrained by the luminosity distance measurement.
grid_sn = contours.frequentist_contour_2d_sparse(
    [likelihoods.DES5yr()],
    grid=param_space,
    fixed=dict(
        fixed,
        H0=parameters.Planck18["H0"],
        Omega_b_h2=parameters.Planck18["Omega_b_h2"],
    ),
)

grid_jla = contours.frequentist_contour_2d_sparse(
    [likelihoods.JLA()],
    grid=param_space,
    fixed=dict(
        fixed,
        H0=parameters.Planck18["H0"],
        Omega_b_h2=parameters.Planck18["Omega_b_h2"],
    ),
)
tools.save(grid_jla, "contour_jla.pkl")

# Compute CMB constraints. The "geometric" summary used in this code
# only constrain the angular distance to the last diffusion surface.,
grid_cmb = contours.frequentist_contour_2d_sparse(
    [likelihoods.Planck2018()],
    grid=param_space,
    fixed=fixed,
)
tools.save(grid_cmb, "contour_planck.pkl")

cmb_bao = contours.frequentist_contour_2d_sparse(
    [likelihoods.Planck2018(), likelihoods.DESIDR2()],
    grid=param_space,
    fixed=fixed,
)
tools.save(cmb_bao, "contour_planck_desi.pkl")

cmb_sn = contours.frequentist_contour_2d_sparse(
    [likelihoods.Planck2018(), likelihoods.JLA()],
    grid=param_space,
    fixed=fixed,
)
tools.save(cmb_bao, "contour_planck_JLA.pkl")


# Plot the result
plt.rc("text", usetex=True)
plt.rc("axes.spines", top=False, right=False, bottom=False, left=False)
display.plot_contours(grid_bao, color=display.color_theme[0], filled=True, label="DESI")
display.plot_contours(
    grid_cmb, color=display.color_theme[1], filled=True, label="Planck"
)
display.plot_contours(grid_jla, color=display.color_theme[2], filled=True, label="JLA")
display.plot_contours(
    cmb_bao, color="red", filled=False, label="Planck+DESI", bestfit=True
)
display.plot_contours(
    cmb_sn, color="green", filled=False, label="Planck+JLA", bestfit=True
)
plt.axhline(-1, color="k", ls=":", lw=0.5)
plt.text(0.48, -1, r"$\Lambda$-CDM", ha="right", va="bottom")
plt.legend(loc="lower right", frameon=False)

plt.show()
