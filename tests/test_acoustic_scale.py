from cosmologix.acoustic_scale import rs, z_star, z_drag, theta_MC, dM, dsound_da_approx
from cosmologix import Planck18
from test_distances import params_to_CAMB, lcdm_deviation
import pyccl as ccl
import jax
import camb
import jax.numpy as jnp


def test_acoustic_scale():
    assert abs(z_star(Planck18) - 1091.95) < 1e-2
    #assert abs(z_drag(Planck18) - 1020.715) < 1e-2
    #assert abs(rs(Planck18, z_star(Planck18)) - 144.7884) < 1e-3
    # According to 10.1051/0004-6361/201833910 (Planck 2018 VI) 100
    # ThetaMC = 1.04089 ± 0.00031 for the base-LCDM bestfit cosmology
    # corresponding to the parameters in Planck18
    assert abs(theta_MC(Planck18) - 1.04089) < 0.0001


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
    from cosmologix.tools import Constants

    #params = lcdm_deviation(m_nu=0)
    params = lcdm_deviation()
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
