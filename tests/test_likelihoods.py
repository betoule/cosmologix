from cosmologix.likelihoods import DES5yr
from cosmologix import Planck18
import jax
from numpy.testing import assert_allclose


def test_sn_likelihood():
    des = DES5yr()
    l = jax.jit(des.negative_log_likelihood)
    params = des.initial_guess(Planck18)
    assert_allclose(l(params), des.negative_log_likelihood(params))
