from cosmologix import distances, contours, display, parameters, likelihoods, fitter
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# Plot settings
plt.rc("text", usetex=True)
plt.rc("axes", **{"spines.top": False, "spines.right": False})
fig = plt.figure(figsize=(10, 6))
axes = fig.subplots(2, 2)
axes[0, 0].set_title("Cosmological distance computations")
axes[0, 1].set_title("Built-in derivatives with jax integration")
axes[1, 0].set_title("Best-fit and fisher matrix analysis")
axes[1, 1].set_title("Frequentist contour exploration")

#
# DISTANCES
#
z_values = jnp.logspace(-2, 3.0, 1000)
axes[0, 0].loglog(
    z_values, distances.dL(parameters.Planck18, z_values), label="luminosity"
)
axes[0, 0].loglog(
    z_values, distances.dM(parameters.Planck18, z_values), label="comoving"
)
axes[0, 0].loglog(
    z_values, distances.dA(parameters.Planck18, z_values), label="angular"
)
axes[0, 0].set_xlabel(r"$z$")
axes[0, 0].set_ylabel(r"$D$ [Mpc]")
axes[0, 0].legend(frameon=False)

#
# DERIVATIVES
#
dmu = jax.jacfwd(distances.mu)
J = dmu(parameters.Planck18.copy(), z_values)
for var in J:
    if var == "Omega_b_h2":
        continue
    axes[0, 1].plot(z_values, J[var], label=display.latex_translation[var])
axes[0, 1].set_yscale("symlog", linthresh=1e-5)
axes[0, 1].set_xscale("log")
axes[0, 1].set_xlabel(r"$z$")
axes[0, 1].set_ylabel(r"$\partial \mu / \partial \theta$")
axes[0, 1].legend(frameon=False, ncols=2)

#
# Bestfit and FIM
#
priors = [likelihoods.Planck2018(), likelihoods.DES5yr()]
fixed = {"Omega_k": 0.0, "m_nu": 0.06, "Neff": 3.046, "Tcmb": 2.7255, "wa": 0.0}

result = fitter.fit(priors, fixed=fixed)
display.pretty_print(result)
display.plot_2d(result, "Omega_bc", "w", ax=axes[1, 0])
axes[1, 0].set_ylim(-1.5, -0.6)
axes[1, 0].set_xlim(0.16, 0.48)
axes[1, 0].set_xlabel(display.latex_translation["Omega_bc"])
axes[1, 0].set_ylabel(display.latex_translation["w"])

# Compute frequentist confidence contours
# The progress bar provides a rough upper bound on computation time because
# the actual size of the explored region is unknown at the start of the calculation.
# Improvements to this feature are planned.
grid0 = contours.frequentist_contour_2d_sparse(
    [priors[0]], grid={"Omega_bc": [0.18, 0.48, 30], "w": [-0.6, -1.5, 30]}, fixed=fixed
)
grid1 = contours.frequentist_contour_2d_sparse(
    [priors[1]],
    grid={"Omega_bc": [0.18, 0.48, 30], "w": [-0.6, -1.5, 30]},
    fixed=dict(
        fixed,
        H0=parameters.Planck18["H0"],
        Omega_b_h2=parameters.Planck18["Omega_b_h2"],
    ),
)
grid = contours.frequentist_contour_2d_sparse(
    priors, grid={"Omega_bc": [0.18, 0.48, 30], "w": [-0.6, -1.5, 30]}, fixed=fixed
)

display.plot_contours(
    grid0, filled=True, ax=axes[1, 1], label="CMB", color=display.color_theme[1]
)
display.plot_contours(
    grid1,
    filled=True,
    ax=axes[1, 1],
    label="DES-5yr",
    color=display.color_theme[2],
)
display.plot_contours(
    grid,
    filled=False,
    label="CMB+DES-5yr",
    ax=axes[1, 1],
    color=display.color_theme[0],
)
axes[1, 1].set_xlabel(display.latex_translation["Omega_bc"])
axes[1, 1].set_ylabel(display.latex_translation["w"])

plt.ion()
plt.legend(loc="lower right", frameon=False)

plt.tight_layout()

plt.savefig("features.svg")
plt.show()
