import jax.numpy as jnp
from jax import lax
import jax
from typing import Callable, Tuple, Dict
from .tools import linear_interpolation, Constants, restrict
from .radiation import Omega_n_mass, Omega_n_rel, Tcmb_to_Omega_gamma


def Omega_c(params: Dict[str, float]) -> float:
    """Compute Omega_c from the total matter density minus baryonic and neutrino contributions."""
    h = params["H0"] / 100
    Omega_b0 = params["Omega_b_h2"] / h**2
    Omega_nu_mass = Omega_n_mass(params, jnp.atleast_1d(1.0))[0]
    return params["Omega_m"] - Omega_b0 - Omega_nu_mass


def Omega_de(params: Dict[str, float], Omega_n_rel: float) -> float:
    """Calculate the dark energy density parameter."""
    omg = Tcmb_to_Omega_gamma(params["Tcmb"], params["H0"])
    return 1 - params["Omega_m"] - omg - Omega_n_rel - params["Omega_k"]


def dzoveru3H(params: Dict[str, float], u: jnp.ndarray) -> jnp.ndarray:
    """Integrand for inverse expansion rate integrals.

    Return (1+z)^{-3/2} H0/H(z)
    """
    h = params["H0"] / 100
    Omega_b0 = params["Omega_b_h2"] / h**2
    Omega_c0 = Omega_c(params)
    Omega_gamma = Tcmb_to_Omega_gamma(params["Tcmb"], params["H0"])
    Omega_nu_rel = Omega_n_rel(params)
    Omega_nu_mass = Omega_n_mass(params, u)
    Omega_de0 = Omega_de(params, Omega_nu_rel)
    return 1.0 / jnp.sqrt(
        (Omega_c0 + Omega_b0)
        + Omega_de0 * u ** (-6 * params["w"])
        + (Omega_gamma + Omega_nu_rel + Omega_nu_mass) * u ** (-2)
        + params["Omega_k"] * u**2
    )


def dC(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the comoving distance at redshift z.

    Distance between comoving object and observer that stay
    constant with time (coordinate).

    Parameters:
    -----------
    params: pytree containing the background cosmological parameters
    z: scalar or array
       redshift at which to compute the comoving distance

    Returns:
    --------
    Comoving distance in Mpc
    """
    dh = Constants.c / params["H0"] * 1e-3  # Hubble radius in kpc
    u = 1 / jnp.sqrt(1 + z)
    umin = 0.02
    step = (1 - umin) / nstep
    _u = jnp.arange(umin + 0.5 * step, 1, step)
    csum = jnp.cumsum(dzoveru3H(params, _u[-1::-1]))[-1::-1]
    return linear_interpolation(u, csum, _u - 0.5 * step) * 2 * step * dh


def dM(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the transverse comoving distance.

    The comoving distance between two comoving objects (distant
    galaxies for examples) separated by an angle theta is dM
    theta.
    """
    comoving_distance = dC(params, z, nstep)
    index = -jnp.sign(params["Omega_k"]).astype(jnp.int8) + 1
    dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in kpc
    sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))

    def open(com_dist):
        return (dh / sqrt_omegak) * jnp.sinh(sqrt_omegak * com_dist / dh)

    def flat(com_dist):
        return com_dist

    def close(com_dist):
        return (dh / sqrt_omegak) * jnp.sin(sqrt_omegak * com_dist / dh)

    branches = (open, flat, close)
    return lax.switch(index, branches, comoving_distance)


def dL(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the luminosity distance in Mpc."""
    return (1 + z) * dM(params, z, nstep)


def dA(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the angular diameter distance in Mpc.

    The physical proper size of a galaxy which subtend an angle
    theta on the sky is dA * theta
    """
    return dM(params, z, nstep) / (1 + z)


def mu(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the distance modulus."""
    return 5 * jnp.log10(dL(params, z, nstep)) + 25


def dVc(params: Dict[str, float], z: jnp.ndarray) -> jnp.ndarray:
    """Calculate the differential comoving volume."""
    dh = Constants.c / params["H0"] * 1e-3
    toto = 1 + z
    return (
        4
        * jnp.pi
        * (dC(params, z) ** 2)
        * dh
        * dzoveru3H(params, 1 / jnp.sqrt(toto))
        / (toto ** (3 / 2))
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from astropy import cosmology
    import numpy as np

    plt.ion()

    #
    # Distance modulus comparison
    #
    Tcmb = 2.7255  # Kelvin, Fixen (2009)
    cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.30, Tcmb0=Tcmb, Neff=0)
    z = jnp.linspace(0.01, 1)
    params = {
        "Omega_m": 0.3,
        "Tcmb": Tcmb,
        "Omega_b_h2": 0.02204854,
        "Omega_k": 0.0,
        "w": -1.0,
        "H0": 70.0,
        "m_nu": 0.06,
        "Neff": 3.046,
    }

    fig = plt.figure("Comparison")
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(z, mu(params, z), label="edris")
    ax1.plot(z, cosmo.distmod(np.asarray(z)).value, label="astropy", ls="--")
    ax1.legend()
    ax2.plot(z, mu(params, z) - cosmo.distmod(np.asarray(z)).value, label="astropy")
    ax2.set_xlabel(r"$z$")
    ax1.set_ylabel(r"$\mu$ [mag]")
    ax2.set_ylabel(r"$\Delta \mu$ [mag]")

    #
    # Gradients
    #
    params["Omega_k"] = 0.001  # Need to work to get dmu/dOmega_k around 0
    plt.figure("Gradients")
    J = jax.jacobian(mu)
    zbig = jnp.linspace(0.01, 1000, 3000)
    G = J(params, zbig)
    for p in params:
        plt.plot(zbig, G[p], label=p)
    plt.yscale("symlog", linthresh=1e-5)
    plt.xscale("log")
    plt.xlabel("z")
    plt.ylabel(r"$\frac{\partial \mu}{\partial \theta}$")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()

    #
    # Acoustic scale
    #
    zstar = z_star(params)
    rsstar = rs(params, zstar)
    thetastar = rsstar / dM(params, zstar) * 100.0
    print(f"{zstar=}, {rsstar=}, {thetastar=}")
    print(f"{planck2018.likelihood(params)}")

    pl = restrict(
        planck2018.likelihood, {"Omega_gamma": Tcmb_to_Omega_gamma(Tcmb, 70), "w": -1}
    )
    x0 = {
        "Omega_m": 0.3,
        "Omega_b_h2": 0.02204854,
        "H0": 70.0,
    }
    # bf18 = tncg(pl, x0, verbose=True)
    import jax

    f = jax.jit(planck2018.weighted_residuals)
    J = jax.jit(jax.jacobian(planck2018.weighted_residuals))
    plt.show()
