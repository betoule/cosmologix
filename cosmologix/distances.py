"""Distance computation facilities."""

from typing import Dict
from functools import partial
import jax.numpy as jnp
from jax import lax
import jax
from cosmologix.densities import Omega
from .tools import Constants

jax.config.update("jax_enable_x64", True)


def distance_integrand(params, u):
    """Integrand for the computation of comoving distance.

    The use of a regular quadrature is possible with the variable change
    u = 1 / sqrt(1+z).

    Args:
        params: A pytree containing the background cosmological parameters.
            Required parameters depend on the cosmological model, but generally
            include H0, Omega_m, Omega_k, etc.
        u: A scalar or jax.numpy.ndarray representing the integration variable,
            where u = 1 / sqrt(1+z).

    Returns:
        The value of the integrand, (1+z)^{-3/2} H0/H(z).
    """
    z = 1 / u**2 - 1
    return 1 / (u**3 * jnp.sqrt(Omega(params, z)))


@partial(jax.jit, static_argnames=("nstep",))
def dC(params, z, nstep=1000):
    """Compute the comoving distance at redshift z.

    Distance between comoving object and observer that stay
    constant with time (coordinate).

    Args:
        params: A pytree containing the background cosmological parameters.
            Must contain 'H0'. Other required parameters depend on the
            cosmological model (e.g., Omega_m, Omega_k).
        z: A scalar or jax.numpy.ndarray of redshift at which to compute the
            comoving distance.
        nstep: An integer representing the number of steps for the integration.

    Returns:
        The comoving distance in Mpc as a jax.numpy.ndarray.
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

    Args:
        params: A pytree containing the background cosmological parameters.
            Must contain 'H0'. Other required parameters depend on the
            cosmological model (e.g., Omega_m, Omega_k).
        z: A scalar or jax.numpy.ndarray of redshift at which to compute the
            lookback time.
        nstep: An integer representing the number of steps for the integration.

    Returns:
        The lookback time in Gyr as a jax.numpy.ndarray.
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
def _transverse_comoving_distance_open(
    params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000
) -> jnp.ndarray:
    """Computes the transverse comoving distance in an open universe.

    Args:
        params: A dictionary of cosmological parameters. Must contain 'H0' and
            'Omega_k'. Other required parameters are inherited from dC.
        z: A jax.numpy.ndarray of redshift values.
        nstep: The number of steps for the integration.

    Returns:
        The transverse comoving distance in Mpc as a jax.numpy.ndarray.
    """
    com_dist = dC(params, z, nstep)
    dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in Mpc
    sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))
    return (dh / sqrt_omegak) * jnp.sinh(sqrt_omegak * com_dist / dh)


@partial(jax.jit, static_argnames=("nstep",))
def _transverse_comoving_distance_closed(
    params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000
) -> jnp.ndarray:
    """Computes the transverse comoving distance in a closed universe.

    Args:
        params: A dictionary of cosmological parameters. Must contain 'H0' and
            'Omega_k'. Other required parameters are inherited from dC.
        z: A jax.numpy.ndarray of redshift values.
        nstep: The number of steps for the integration.

    Returns:
        The transverse comoving distance in Mpc as a jax.numpy.ndarray.
    """
    com_dist = dC(params, z, nstep)
    dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in Mpc
    sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))
    return (dh / sqrt_omegak) * jnp.sin(sqrt_omegak * com_dist / dh)


@partial(jax.jit, static_argnames=("nstep",))
def dM(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the transverse comoving distance in Mpc.

    Args:
        params: A dictionary of cosmological parameters. Must contain 'Omega_k'.
            Other required parameters are inherited from the called functions.
        z: A jax.numpy.ndarray of redshift values.
        nstep: The number of steps for the integration.

    Returns:
        The transverse comoving distance in Mpc as a jax.numpy.ndarray.

    Notes:
        The default quadrature uses 1000 steps which does not provide
        10⁻⁴ accuracy below z<0.01 (see
        https://lemaitre.pages.in2p3.fr/cosmologix/numerical_results.html).
        If your use case requires high-accuracy at lower redshift
        consider using nstep=10000 instead.
    """
    index = -jnp.sign(params["Omega_k"]).astype(jnp.int8) + 1
    # we need to pass nstep explicitly to branches to avoid
    # lax.switch’s dynamic argument passing
    return lax.switch(
        index,
        [
            lambda p, z: _transverse_comoving_distance_open(p, z, nstep),
            lambda p, z: dC(p, z, nstep),
            lambda p, z: _transverse_comoving_distance_closed(p, z, nstep),
        ],
        params,
        z,
    )


def dL(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the luminosity distance in Mpc.

    Args:
        params: A dictionary of cosmological parameters, as required by dM.
        z: A jax.numpy.ndarray of redshift values.
        nstep: The number of steps for the integration.

    Returns:
        The luminosity distance in Mpc as a jax.numpy.ndarray.

    Notes:
        The default quadrature uses 1000 steps which does not provide
        10⁻⁴ accuracy below z<0.01 (see
        https://lemaitre.pages.in2p3.fr/cosmologix/numerical_results.html).
        If your use case requires high-accuracy at lower redshift
        consider using nstep=10000 instead.
    """
    return (1 + z) * dM(params, z, nstep)


def dA(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the angular diameter distance in Mpc.

    The physical proper size of a galaxy which subtend an angle
    theta on the sky is dA * theta.

    Args:
        params: A dictionary of cosmological parameters, as required by dM.
        z: A jax.numpy.ndarray of redshift values.
        nstep: The number of steps for the integration.

    Returns:
        The angular diameter distance in Mpc as a jax.numpy.ndarray.

    Notes:
        The default quadrature uses 1000 steps which does not provide
        10⁻⁴ accuracy below z<0.01 (see
        https://lemaitre.pages.in2p3.fr/cosmologix/numerical_results.html).
        If your use case requires high-accuracy at lower redshift
        consider using nstep=10000 instead.
    """
    return dM(params, z, nstep) / (1 + z)


def dH(params: Dict[str, float], z: jnp.ndarray) -> jnp.ndarray:
    """Compute the Hubble distance in Mpc.

    Args:
        params: A dictionary of cosmological parameters, as required by H.
        z: A jax.numpy.ndarray of redshift values.

    Returns:
        The Hubble distance in Mpc as a jax.numpy.ndarray.
    """
    return Constants.c * 1e-3 / H(params, z)


def H(params: Dict[str, float], z: jnp.ndarray) -> jnp.ndarray:
    """Hubble rate in km/s/Mpc.

    Args:
        params: A pytree containing the background cosmological parameters.
            Must contain 'H0'. Other required parameters are inherited from Omega.
        z: A scalar or jax.numpy.ndarray of redshift at which to compute the
            Hubble rate.

    Returns:
        The Hubble rate in km/s/Mpc as a jax.numpy.ndarray.
    """
    return params["H0"] * jnp.sqrt(Omega(params, z))


@partial(jax.jit, static_argnames=("nstep",))
def mu(params: Dict[str, float], z: jnp.ndarray, nstep: int = 1000) -> jnp.ndarray:
    """Compute the distance modulus.

    Args:
        params: A dictionary of cosmological parameters, as required by dL.
        z: A jax.numpy.ndarray of redshift values.
        nstep: The number of steps for the integration.

    Returns:
        The distance modulus as a jax.numpy.ndarray.

    Notes:
        The default quadrature uses 1000 steps which does not provide
        10⁻⁴ accuracy below z<0.01 (see
        https://lemaitre.pages.in2p3.fr/cosmologix/numerical_results.html).
        If your use case requires high-accuracy at lower redshift
        consider using nstep=10000 instead.
    """
    return 5 * jnp.log10(dL(params, z, nstep)) + 25


def dV(params: Dict[str, float], z: jnp.ndarray) -> jnp.ndarray:
    """Calculate the volumic distance.

    See formula 2.6 in DESI 1yr cosmological results arxiv:2404.03002.

    Args:
        params: A dictionary of cosmological parameters, as required by dM and dH.
        z: A jax.numpy.ndarray of redshift values.

    Returns:
        The volumic distance as a jax.numpy.ndarray.
    """
    return (z * dM(params, z) ** 2 * dH(params, z)) ** (1 / 3)


def _flat_comoving_volume(params, z):
    """Computes the comoving volume in a flat universe.

    Args:
        params: A dictionary of cosmological parameters, as required by dC.
        z: A jax.numpy.ndarray of redshift values.

    Returns:
        The comoving volume in a flat universe as a jax.numpy.ndarray.
    """
    return 1.0 / 3.0 * (dC(params, z) ** 3)


def _open_comoving_volume(params, z):
    """Computes the comoving volume in an open universe.

    Args:
        params: A dictionary of cosmological parameters. Must contain 'H0' and
            'Omega_k'. Other required parameters are inherited from dC.
        z: A jax.numpy.ndarray of redshift values.

    Returns:
        The comoving volume in an open universe as a jax.numpy.ndarray.
    """
    comoving_coordinate = dC(params, z)
    dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in Mpc
    sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))
    comoving_distance = (dh / sqrt_omegak) * jnp.sinh(
        sqrt_omegak * comoving_coordinate / dh
    )
    d = comoving_distance / dh
    return (
        dh**2
        / (2.0 * params["Omega_k"])
        * (
            comoving_distance * jnp.sqrt(1 + params["Omega_k"] * d**2)
            - comoving_coordinate
        )
    )


def _close_comoving_volume(params, z):
    """Computes the comoving volume in a closed universe.

    Args:
        params: A dictionary of cosmological parameters. Must contain 'H0' and
            'Omega_k'. Other required parameters are inherited from dC.
        z: A jax.numpy.ndarray of redshift values.

    Returns:
        The comoving volume in a closed universe as a jax.numpy.ndarray.
    """
    comoving_coordinate = dC(params, z)
    dh = Constants.c / params["H0"] * 1e-3  # Hubble distance in Mpc
    sqrt_omegak = jnp.sqrt(jnp.abs(params["Omega_k"]))
    comoving_distance = (dh / sqrt_omegak) * jnp.sin(
        sqrt_omegak * comoving_coordinate / dh
    )
    d = comoving_distance / dh
    return (
        dh**2
        / (2.0 * params["Omega_k"])
        * (
            comoving_distance * jnp.sqrt(1 + params["Omega_k"] * d**2)
            - comoving_coordinate
        )
    )


def comoving_volume(
    params: Dict[str, float], z: jnp.ndarray, solid_angle: float = 4 * jnp.pi
) -> jnp.ndarray:
    """Compute the comoving volume for given redshifts range and solid angle.

    Args:
        params: A dictionary of cosmological parameters. Must contain 'Omega_k'.
            Other required parameters are inherited from the called functions.
        z: A jax.numpy.ndarray of redshift values at which to compute the
            comoving volume.
        solid_angle: The solid angle in steradians over which the volume is
            calculated. Defaults to 4π, corresponding to the full sky.

    Returns:
        A jax.numpy.ndarray of comoving volumes in cubic megaparsecs (Mpc³)
        corresponding to each redshift in `z`, scaled by the specified solid
        angle.
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

    Args:
        params: A dictionary of cosmological parameters, as required by dM and H.
        z: A jax.numpy.ndarray of redshift values at which to compute the
            differential volume.

    Returns:
        A jax.numpy.ndarray of differential comoving volume elements in cubic
        megaparsecs per unit redshift per steradian (Mpc³/sr/z) at each
        redshift in `z`.
    """
    return Constants.c * 1e-3 * dM(params, z) ** 2 / H(params, z)
