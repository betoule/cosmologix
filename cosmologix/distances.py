import jax.numpy as jnp
from jax import lax
import jax
from typing import Callable, Tuple, Dict
from .tools import Constants
from .interpolation import linear_interpolation
from cosmologix.densities import Omega
from functools import partial

jax.config.update("jax_enable_x64", True)


def distance_integrand(params, u):
    """Integrand for the computation of comoving distance

    The use of a regular quadradure is possible with the variable change
    u = 1 / sqrt(1+z)


    the function return (1+z)^{-3/2} H0/H(z).
    """
    z = 1 / u**2 - 1
    return 1 / (u**3 * jnp.sqrt(Omega(params, z)))


@partial(jax.jit, static_argnames=("nstep",))
def dC(params, z, nstep=1000):
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
    dh = Constants.c / params["H0"] * 1e-3  # in Mpc
    u = 1 / jnp.sqrt(1 + z)
    umin = 0.02
    step = (1 - umin) / nstep
    _u = jnp.arange(umin + 0.5 * step, 1, step)
    csum = jnp.cumsum(distance_integrand(params, _u[-1::-1]))[-1::-1]
    return jnp.interp(u, _u - 0.5 * step, csum) * 2 * step * dh


@partial(jax.jit, static_argnames=("nstep",))
def lookback_time(params, z, nstep=1000):
    """Compute the lookback time at redshift z.

    Parameters:
    -----------
    params: pytree containing the background cosmological parameters
    z: scalar or array
       redshift at which to compute the lookback time

    Returns:
    --------
    Lookback time in Gyr
    """
    costime = (
        1 / (params["H0"] / (Constants.pc * 1e6 / 1e3)) / Constants.year / 1e9
    )  # Gyr
    u = 1 / jnp.sqrt(1 + z)
    umin = 0.02
    step = (1 - umin) / nstep
    _u = jnp.arange(umin + 0.5 * step, 1, step)
    csum = jnp.cumsum(_u[-1::-1] ** 2 * distance_integrand(params, _u[-1::-1]))[-1::-1]
    return jnp.interp(u, _u - 0.5 * step, csum) * 2 * step * costime


@partial(jax.jit, static_argnames=("nstep",))
def _dM_open(
    params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000
) -> jnp.ndarray:
    com_dist = dC(params, z, nstep)
    dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in Mpc
    sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))
    return (dh / sqrt_omegak) * jnp.sinh(sqrt_omegak * com_dist / dh)


@partial(jax.jit, static_argnames=("nstep",))
def _dM_close(
    params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000
) -> jnp.ndarray:
    com_dist = dC(params, z, nstep)
    dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in Mpc
    sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))
    return (dh / sqrt_omegak) * jnp.sin(sqrt_omegak * com_dist / dh)


@partial(jax.jit, static_argnames=("nstep",))
def dM(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    index = -jnp.sign(params["Omega_k"]).astype(jnp.int8) + 1
    # we need to pass nstep explicitly to branches to avoid
    # lax.switch’s dynamic argument passing
    return lax.switch(
        index,
        [
            lambda p, z: _dM_open(p, z, nstep),
            lambda p, z: dC(p, z, nstep),
            lambda p, z: _dM_close(p, z, nstep),
        ],
        params,
        z,
    )


def dL(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the luminosity distance in Mpc."""
    return (1 + z) * dM(params, z, nstep)
    # return (1 + z) * dM(params, z, nstep)
    # return (1 + z) * dC(params, z, nstep)


def dA(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the angular diameter distance in Mpc.

    The physical proper size of a galaxy which subtend an angle
    theta on the sky is dA * theta
    """
    return dM(params, z, nstep) / (1 + z)


def dH(params: Dict[str, float], z: jnp.ndarray) -> jnp.ndarray:
    """Compute the Hubble distance in Mpc."""
    return Constants.c * 1e-3 / H(params, z)


def H(params: Dict[str, float], z: jnp.ndarray) -> jnp.ndarray:
    """Hubble rate in km/s/Mpc.

    Parameters:
    -----------
    params: pytree containing the background cosmological parameters
    z: scalar or array
       redshift at which to compute the comoving distance


    u = 1/sqrt(1+z)

    """
    return params["H0"] * jnp.sqrt(Omega(params, z))

@partial(jax.jit, static_argnames=("nstep",))
def mu(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the distance modulus."""
    return 5 * jnp.log10(dL(params, z, nstep)) + 25


def dVc(params: Dict[str, float], z: jnp.ndarray) -> jnp.ndarray:
    """Calculate the differential comoving volume."""
    dh = Constants.c / params["H0"] * 1e-3
    u = 1.0 / jnp.sqrt(1 + z)
    return 4 * jnp.pi * (dC(params, z) ** 2) * dh * dzoveru3H(params, u) * u**3


def dV(params: Dict[str, float], z: jnp.ndarray) -> jnp.ndarray:
    """Calculate the volumic distance.
    See formula 2.6 in DESI 1yr cosmological results arxiv:2404.03002
    """
    return (z * dM(params, z) ** 2 * dH(params, z)) ** (1 / 3)


def _flat_comoving_volume(params, z):
    return 1.0 / 3.0 * (dC(params, z) ** 3)


def _open_comoving_volume(params, z):
    _dC = dC(params, z)
    dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in Mpc
    sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))
    _dM = (dh / sqrt_omegak) * jnp.sinh(sqrt_omegak * _dC / dh)
    d = _dM / dh
    return (
        dh**2
        / (2.0 * params["Omega_k"])
        * (_dM * jnp.sqrt(1 + params["Omega_k"] * d**2) - _dC)
    )


def _close_comoving_volume(params, z):
    _dC = dC(params, z)
    dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in Mpc
    sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))
    _dM = (dh / sqrt_omegak) * jnp.sin(sqrt_omegak * _dC / dh)
    d = _dM / dh
    return (
        dh**2
        / (2.0 * params["Omega_k"])
        * (_dM * jnp.sqrt(1 + params["Omega_k"] * d**2) - _dC)
    )


def comoving_volume(
    params: Dict[str, float], z: jnp.ndarray, solid_angle: float = 4 * jnp.pi
) -> jnp.ndarray:
    """Compute the comoving volume for given redshifts range and solid angle.

    Parameters
    ----------
    params : dict
        Dictionary of cosmological parameters, including 'Omega_k' (curvature parameter),
        and others required by the volume computation functions (e.g., 'H0', 'Omega_m').
    z : jax.numpy.ndarray
        Array of redshift values at which to compute the comoving volume.
    solid_angle : float, optional
        Solid angle in steradians over which the volume is calculated (default: 4π,
        corresponding to the full sky).

    Returns
    -------
    jax.numpy.ndarray
        Array of comoving volumes in cubic megaparsecs (Mpc³) corresponding to each
        redshift in `z`, scaled by the specified solid angle.
    """
    index = -jnp.sign(params["Omega_k"]).astype(jnp.int8) + 1
    return solid_angle * lax.switch(
        index,
        [_open_comoving_volume, _flat_comoving_volume, _close_comoving_volume],
        params,
        z,
    )


def differential_comoving_volume(
    params: Dict[str, float], z: jnp.ndarray
) -> jnp.ndarray:
    """Compute the differential comoving volume element per unit redshift and steradian.

    This function calculates dV_c/dz, the differential comoving volume element, which
    is used to determine the volume of a spherical shell at a given redshift in a
    cosmological model.

    Parameters
    ----------
    params : dict
        Dictionary of cosmological parameters, including those needed for the Hubble
        parameter 'H' and comoving distance 'dM' (e.g., 'H0', 'Omega_m', 'Omega_k').
    z : jax.numpy.ndarray
        Array of redshift values at which to compute the differential volume.

    Returns
    -------
    jax.numpy.ndarray
        Array of differential comoving volume elements in cubic megaparsecs per unit
        redshift per steradian (Mpc³/sr/z) at each redshift in `z`.
    """
    return Constants.c * 1e-3 * dM(params, z) ** 2 / H(params, z)
