from cosmologix import neutrinos
import jax.numpy as jnp


def test_composition(plot=False):
    mbar = jnp.logspace(-3, 4, 1000)
    I_composite = neutrinos.compute_composite_integral(mbar)
    I_numeric = neutrinos.compute_fermion_distribution_integral(mbar)
    if plot:
        import matplotlib.pyplot as plt

        fig = plt.figure("Integral_accuracy")
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        ax1.loglog(mbar, I_numeric, "k", label="numeric")
        ax1.loglog(mbar, I_composite, "r--", label="composite")
        ax1.legend(loc="best")
        ax2.axhline(0, color="k")
        ax2.plot(mbar, I_composite / I_numeric - 1, "r")
        plt.show()
    assert (jnp.abs(I_composite / I_numeric - 1) < 1e-5).all()


# _mbar = jnp.logspace(-2, 2, 1000)
# _Imbar = I_m(_mbar)
#
#
# def interpolated_I(m_bar):
#    return linear_interpolation(jnp.log(m_bar), _Imbar, jnp.log(_mbar))


if __name__ == "__main__":
    from cosmologix import Planck18
    import matplotlib.pyplot as plt

    plt.ion()
    test_composition(True)

    # params = densities.params_to_density_params(Planck18.copy())

    # mbar = jnp.logspace(-3, 4, 1000)
    # mlim = jnp.logspace(-2, 3, 1000)
    #
    # plt.figure("neutrinos density")
    # z = jnp.linspace(0.01, 1000, 1000)
    # plt.plot(z, rho_nu(params, z))
    # plt.show()
