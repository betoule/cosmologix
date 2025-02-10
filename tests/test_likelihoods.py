from cosmologix.likelihoods import DES5yr, Planck2018Prior, LikelihoodSum
from cosmologix import Planck18
from cosmologix.fitter import unflatten_vector, flatten_vector
import jax
from numpy.testing import assert_allclose
import time

jax.config.update("jax_enable_x64", True)


def test_likelihoods():
    des = DES5yr()
    pl = Planck2018Prior()
    both = LikelihoodSum([des, pl])
    for likelihood in [both]:
        l = jax.jit(likelihood.negative_log_likelihood)
        params = likelihood.initial_guess(Planck18)
        assert_allclose(l(params), likelihood.negative_log_likelihood(params))
        assert_allclose(l(params), likelihood.negative_log_likelihood(params))
        x = flatten_vector(params)
        _l = lambda x: l(unflatten_vector(params, x))
        g = jax.grad(_l)
        gj = jax.jit(g)
        assert_allclose(g(x), gj(x))
        assert_allclose(g(x), gj(x))

        H = jax.hessian(_l)
        Hj = jax.jit(H)
        assert_allclose(H(x), Hj(x))
        assert_allclose(H(x), Hj(x))


def toto():
    des = DES5yr()
    pl = Planck2018Prior()
    params = des.initial_guess(Planck18)
    des.negative_log_likelihood(params)
    pl.negative_log_likelihood(params)
    l1 = jax.jit(des.negative_log_likelihood)
    l2 = jax.jit(pl.negative_log_likelihood)
    l1(params)
    l2(params)
    l1(params)
    l2(params)
