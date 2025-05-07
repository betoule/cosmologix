from cosmologix.acoustic_scale import (
    rd,
    rs,
    z_star,
    z_drag,
    theta_MC,
    dM,
    dsound_da_approx,
    rd_approx,
)
from cosmologix import Planck18, densities
from cosmologix.parameters import DESI2024YR1_Fiducial
from cosmologix.tools import Constants
from test_distances import params_to_CAMB, lcdm_deviation
import pyccl as ccl
import jax
import camb
import jax.numpy as jnp


def test_acoustic_scale():
    params = densities.process_params(Planck18)
    assert abs(z_star(params) - 1091.95) < 1e-2
    # The following are not really used in practice, still testing
    # them as they are provided for convenience.
    assert abs(z_drag(params) - 1020.55) < 1e-2
    assert abs(rs(params, z_star(params)) - 144.34) < 1e-2
    assert abs(rd(params) - 150.84) < 1e-2
    # What matters to contour accuracy is theta_MC and rd_approx

    # According to 10.1051/0004-6361/201833910 (Planck 2018 VI) 100
    # ThetaMC = 1.04089 ± 0.00031 for the base-LCDM bestfit cosmology
    # corresponding to the parameters in Planck18
    assert abs(theta_MC(params) - 1.04089) < 0.0001
    # Computation made by dividing alpha by reported distance in
    # arxiv/2404.03000
    assert abs(rd_approx(DESI2024YR1_Fiducial) - 147.238) < 1e-2


def timings():
    zs = jax.jit(z_star)
    zd = jax.jit(z_drag)
    rsj = jax.jit(rs)
    zs(Planck18)
    zd(Planck18)
    rsj(Planck18, zs(Planck18))
    zs(Planck18)
    zd(Planck18)
    rsj(Planck18, zs(Planck18))


if __name__ == "__main__":
    # params = lcdm_deviation(m_nu=0)
    params = densities.process_params(lcdm_deviation())
    pars = params_to_CAMB(params)
    zstar = z_star(params)
    astar = 1 / (1 + zstar)
    results = camb.get_results(pars)
    thetastar = theta_MC(params)
    print(f"CAMB: {100*results.cosmomc_theta()}")
    print(f"Cosmologix: {thetastar}")
    print(Constants.c * 1e-3 / jnp.sqrt(3) * dsound_da_approx(params, 1e-8))
    print(Constants.c * 1e-3 / jnp.sqrt(3) * dsound_da_approx(params, astar / 2))
    print(Constants.c * 1e-3 / jnp.sqrt(3) * dsound_da_approx(params, astar))
