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

@partial(jax.jit, static_argnames=('nstep',))
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


@partial(jax.jit, static_argnames=('nstep',))
def _dM_open(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    com_dist = dC(params, z, nstep)
    dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in Mpc
    sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))
    return (dh / sqrt_omegak) * jnp.sinh(sqrt_omegak * com_dist / dh)

@partial(jax.jit, static_argnames=('nstep',))
def _dM_close(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    com_dist = dC(params, z, nstep)
    dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in Mpc
    sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))
    return (dh / sqrt_omegak) * jnp.sin(sqrt_omegak * com_dist / dh)

@partial(jax.jit, static_argnames=('nstep',))
def dM_static(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    index = -jnp.sign(params["Omega_k"]).astype(jnp.int8) + 1
    return lax.switch(index, 
                      [lambda p, z: _dM_open(p, z, nstep),
                       lambda p, z: dC(p, z, nstep),
                       lambda p, z: _dM_close(p, z, nstep)],
                      params, z)

#@partial(jax.jit, static_argnames=('nstep',))
#@jax.jit
#def dM_static(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
##def dM_static(params: Dict[str, float], z: jnp.ndarray) -> jnp.ndarray:
#    index = -jnp.sign(params["Omega_k"]).astype(jnp.int8) + 1
#    branches = (_dM_open, dC, _dM_close)
#    return lax.switch(index, branches, params, z)
    
def dM(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the transverse comoving distance.

    The comoving distance between two comoving objects (distant
    galaxies for examples) separated by an angle theta is dM
    theta.
    """
    comoving_distance = dC(params, z, nstep)
    #index = 2-jnp.digitize(params["Omega_k"], jnp.array([-1e-5, 1e-5]))
    #print(index)
    index = -jnp.sign(params["Omega_k"]).astype(jnp.int8) + 1

    def open(com_dist):
        dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in Mpc
        sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))
        return (dh / sqrt_omegak) * jnp.sinh(sqrt_omegak * com_dist / dh)

    def flat(com_dist):
        return com_dist

    def close(com_dist):
        dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in Mpc
        sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))
        return (dh / sqrt_omegak) * jnp.sin(sqrt_omegak * com_dist / dh)

    branches = (open, flat, close)
    return lax.switch(index, branches, comoving_distance)


def dL(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the luminosity distance in Mpc."""
    return (1 + z) * dM_static(params, z, nstep)
    #return (1 + z) * dM(params, z, nstep)
    #return (1 + z) * dC(params, z, nstep)


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
    return (z * dM_static(params, z) ** 2 * dH(params, z)) ** (1 / 3)
