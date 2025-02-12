from cosmologix.likelihoods import DES5yr, LikelihoodSum, Planck2018Prior
from cosmologix import Planck18
from cosmologix.fitter import newton, flatten_vector, unflatten_vector, restrict_to
import jax.numpy as jnp
import jax
import time


jax.config.update("jax_enable_x64", True)


def control_fitter_bias_and_coverage(likelihoods, point, fitter, ndraw=50):
    # Simulated data
    params = Planck18.copy()
    params.update(point)
    likelihood = LikelihoodSum(likelihoods)

    def draw():
        likelihood.draw(params)
        loss, start = restrict_to(
            likelihood.negative_log_likelihood, params, list(point.keys()), flat=False
        )
        bestfit, extra = newton(loss, start)
        return flatten_vector(bestfit)

    results = jnp.array([draw() for _ in range(ndraw)])
    bias = jnp.mean(results, axis=0) - flatten_vector(point)
    sigma = jnp.std(results, axis=0) / jnp.sqrt(ndraw)
    assert (jnp.abs(bias / sigma) < 3).all()


def test_newton_fitter():
    des = DES5yr()
    point = {"Omega_m": 0.3, "M": 0.0}
    control_fitter_bias_and_coverage([des], point, newton, ndraw=50)


def test_fit():
    from cosmologix import likelihoods, fit

    priors = [likelihoods.Planck2018Prior()]
    fixed = {"Omega_k": 0.0, "m_nu": 0.06, "Neff": 3.046, "Tcmb": 2.7255, "w": -1.0}
    result = fit(priors, fixed=fixed)
    print(result["bestfit"])


if __name__ == "__main__":
    des = DES5yr()
    pl = Planck2018Prior()
    point = {
        "Omega_m": 0.3,
        "M": 0,
    }
    #'M': 0.}
    # control_fitter_bias_and_coverage([des, pl], point, newton, ndraw=50)
    test_fit()

#    fixed_params = Planck18.copy()
#    fixed_params.pop("Omega_m")
#    fixed_params.pop("w")
#
#    likelihoods = [des]
#
#
#    starting_point = Planck18
#
