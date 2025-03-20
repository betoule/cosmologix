from cosmologix import Planck18, likelihoods, tools
from cosmologix.fitter import unflatten_vector, flatten_vector
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose
import time

jax.config.update("jax_enable_x64", True)


def func_and_derivatives(func, x, jac=False, hessian=False, funcname=""):
    """Test whether jitted and normal version of a function and its derivative are correct"""
    funcs = {"func": func}
    if jac:
        funcs["jac"] = jax.jacfwd(func)
    elif hessian:
        funcs["grad"] = jax.grad(func)
        funcs["hessian"] = jax.hessian(func)
    for label, f in funcs.items():
        print(f"testing {label} of {funcname}")
        fj = jax.jit(f)
        a = f(x)
        b = fj(x)
        assert jnp.isfinite(a).all(), f"{label} of {funcname} as non finite values"
        assert_allclose(a, b, rtol=1e-7, atol=1e-10)


def get_like_func(likelihood, fix=["Omega_k"]):
    params = likelihood.initial_guess(Planck18.copy())
    fixed = dict([(p, params.pop(p)) for p in fix])
    x = flatten_vector(params)

    R = lambda x: likelihood.weighted_residuals(
        dict(unflatten_vector(params, x), **fixed)
    )
    l = lambda x: likelihood.negative_log_likelihood(
        dict(unflatten_vector(params, x), **fixed)
    )
    return x, l, R


def test_likelihoods(fix=["Omega_k"]):
    tools.clear_cache()
    priors = {
        "desiu": likelihoods.DESIDR1Prior(True),
        "desi": likelihoods.DESIDR1Prior(),
        "des": likelihoods.DES5yr(),
        "union3": likelihoods.Union3(),
        "pantheon+": likelihoods.Pantheonplus(),
        "planck": likelihoods.Planck2018Prior(),
        "jla": likelihoods.JLA(),
        "BBN": likelihoods.BBNSchoneberg2024Prior(),
        "BBNNeff": likelihoods.BBNNeffSchoneberg2024Prior(),
        "SH0ES": likelihoods.SH0ES(),
    }
    priors["sum"] = likelihoods.LikelihoodSum(list(priors.values()))
    for name, likelihood in priors.items():
        x, l, R = get_like_func(likelihood, fix=fix)
        func_and_derivatives(R, x, jac=True, funcname=f"{name}.wres")
        func_and_derivatives(l, x, funcname=f"{name}.likelihood")


if __name__ == "__main__":
    # test_likelihoods()
    priors = {
        "desiu": likelihoods.DESIDR1Prior(True),
        "desi": likelihoods.DESIDR1Prior(),
        "desi2u": likelihoods.DESIDR2Prior(True),
        "desi2": likelihoods.DESIDR2Prior(),
        "des": likelihoods.DES5yr(),
        "union3": likelihoods.Union3(),
        "pantheon+": likelihoods.Pantheonplus(),
        "planck": likelihoods.Planck2018Prior(),
        "jla": likelihoods.JLA(),
        "BBN": likelihoods.BBNSchoneberg2024Prior(),
        "BBNNeff": likelihoods.BBNNeffSchoneberg2024Prior(),
    }
    priors["sum"] = likelihoods.LikelihoodSum([priors["planck"], priors["des"]])
    from cosmologix.tools import speed_measurement

    for name, likelihood in priors.items():
        params = likelihood.initial_guess(Planck18.copy())
        # print(name, speed_measurement(likelihood.negative_log_likelihood, params))
        print(name, speed_measurement(likelihood.weighted_residuals, params))
