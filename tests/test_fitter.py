from cosmologix.likelihoods import DES5yr, LikelihoodSum
from cosmologix import Planck18
from cosmologix.fitter import newton, flatten_vector, unflatten_vector
from cosmologix.tools import randn
import jax.numpy as jnp
import jax
import time


jax.config.update("jax_enable_x64", True)

def restrict(func, complete, varied, flat=True):
    fixed = complete.copy()
    for p in varied:
        fixed.pop(p)
    if flat:
        return lambda x: func(dict(unflatten_vector(varied, x), fixed))
    else:
        return lambda x: func(dict(x, **fixed))

def control_fitter_bias_and_coverage(likelihoods, point, fitter, ndraw=50):
    # Simulated data
    params = Planck18.copy()
    params.update(point)

    def draw():
        for like in likelihoods:
            like.data = like.model(params) + randn(like.error)
        likelihood = LikelihoodSum(likelihoods)
        loss = restrict(likelihood.negative_log_likelihood, params, point, flat=False)
        bestfit, extra = newton(loss, point)
        return flatten_vector(bestfit)
    results = jnp.array([draw() for _ in range(ndraw)])
    bias = jnp.mean(results, axis=0) - flatten_vector(point)
    sigma = jnp.std(results, axis=0) / jnp.sqrt(ndraw)
    assert (jnp.abs(bias / sigma) < 3).all()

def test_newton_fitter():
    des = DES5yr()
    point = {'Omega_m': 0.3,
             'M': 0.}
    control_fitter_bias_and_coverage([des], point, newton, ndraw=50)


if __name__ == '__main__':
    des = DES5yr()
    point = {'Omega_m': 0.3,
             'M': 0.}
    test_newton_fitter([des], point, newton, ndraw=50)
    

#    fixed_params = Planck18.copy()
#    fixed_params.pop("Omega_m")
#    fixed_params.pop("w")
#
#    likelihoods = [des]
#
#    grid={'Omega_m': [0.18,0.48,30],
#          'w': [-0.6, -1.5,30]}
#
#    starting_point = Planck18
#
    
