from accuracy_plots import *
import time
import jax
import numpy as np


def speed_measurement(func, params, z, n=10):
    tstart = time.time()
    result = func(params, z)
    tcomp = time.time()
    dur = []
    for _ in range(n):
        t0 = time.time()
        result = jax.block_until_ready(func(params, z))
        dur.append(time.time() - t0)
    # return (len(z), tcomp-tstart, (tstop-tcomp)/n)
    return (tcomp - tstart, np.mean(dur), np.std(dur) / np.sqrt(n))


if __name__ == "__main__":
    plt.ion()
    plt.rc("text", usetex=True)
    plt.rc("axes.spines", top=False, right=False, bottom=True, left=True)

    tested = {
        # "cosmologix": mu,
        # "cosmologix (jit)": jax.jit(mu),
        # "cosmologix (grad)": jax.jacfwd(mu),
        # "cosmologix (grad-jit)": jax.jit(jax.jacfwd(mu)),
        r"cosmologix ($\mu$)": jax.jit(mu),
        r"cosmologix ($\vec \nabla \mu$)": jax.jit(jax.jacfwd(mu)),
        "ccl": mu_ccl,
        "camb": mu_camb,
        "astropy": mu_astropy,
        "jax_cosmo": mu_jaxcosmo,
        "cosmoprimo": mu_cosmoprimo,
    }
    params = Planck18.copy()
    ns = jnp.array([10, 30, 100, 300, 1000, 3000, 10000])
    result = {}
    for func in tested:
        result[func] = jnp.array(
            [
                speed_measurement(tested[func], params, np.linspace(0.01, 1, n))
                for n in ns
            ]
        )

    fig = plt.figure("mu_speed")
    ax1, ax2 = fig.subplots(1, 2, sharey=True, sharex=True)
    for func in tested:
        ax2.errorbar(ns, result[func][:, 1], result[func[:, 2]], label=func)
    for func in tested:
        ax1.plot(ns, result[func][:, 0], label=func)
    ax1.set_title("first call")
    ax2.set_title("subsequent calls")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("len(z)")
    ax2.set_xlabel("len(z)")
    ax1.set_ylabel("wall time [s]")
    ax1.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig("doc/mu_speed.pdf")
