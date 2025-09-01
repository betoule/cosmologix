from cosmologix import (
    likelihoods,
    distances,
    acoustic_scale,
    display,
)
from cosmologix.parameters import get_cosmo_params
from cosmologix.fitter import flatten_vector, unflatten_vector, restrict_to, fit
import jax.numpy as jnp
import jax
import time
import pytest

jax.config.update("jax_enable_x64", True)


def control_fitter_bias_and_coverage(priors, point, fitter, ndraw=50):
    # Simulated data
    params = get_cosmo_params()
    params.update(point)
    likelihood = likelihoods.LikelihoodSum(priors)

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


# @pytest.mark.slow
# def test_newton_fitter():
#    des = likelihoods.DES5yr()
#    point = {"Omega_bc": 0.3, "M": 0.0}
#    control_fitter_bias_and_coverage([des], point, newton, ndraw=50)


def test_simple_fit():
    priors = [likelihoods.Planck2018()]
    fixed = {
        "Omega_k": 0.0,
        "m_nu": 0.06,
        "Neff": 3.046,
        "Tcmb": 2.7255,
        "w": -1.0,
        "wa": 0.0,
    }
    result = fit(priors, fixed=fixed, verbose=True)
    display.pretty_print(result)
    display.plot_2d(result, "Omega_bc", "Omega_b_h2")


def test_de_fit():
    priors = [
        likelihoods.Planck2018(),
        likelihoods.DES5yr(),
        likelihoods.DESIDR1(),
    ]
    fixed = {
        "Omega_k": 0.0,
        "m_nu": 0.06,
        "Neff": 3.046,
        "Tcmb": 2.7255,
    }
    result = fit(priors, fixed=fixed, verbose=True)
    display.pretty_print(result)
    display.plot_2d(result, "w", "wa")


if __name__ == "__main__":
    des = likelihoods.DES5yr()
    pl = likelihoods.Planck2018()
    desiu = likelihoods.DESIDR1(True)
    point = {
        "Omega_bc": 0.3,
        "M": 0.0,
    }


#    fixed_params = Planck18.copy()
#    fixed_params.pop("Omega_bc")
#    fixed_params.pop("w")
#
#    likelihoods = [des]
#
#
#    starting_point = Planck18
#
