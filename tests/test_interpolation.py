import jax
import jax.numpy as jnp
import cosmologix.interpolation
from cosmologix.tools import speed_measurement


def jnp_interpolant(func, a, b, n=10):
    nodes = jnp.linspace(a, b, n)
    return lambda x: jnp.interp(x, nodes, func(nodes))


def interpolation_accuracy(func, method, a=0, b=1, n=10):
    interp = method(func, a, b, n)
    x = jnp.linspace(a, b, 4000)
    y = func(x)
    yi = interp(x)
    speed = speed_measurement(interp, x)
    results = {
        "first_call": speed[0],
        "subsequent": speed[1],
        "first_call(j)": speed[3],
        "subsequent(j)": speed[4],
        "accuracy": (yi - y).ptp(),
    }
    return results


def test_interpolation():
    def x2(x):
        return x**2

    for func in [x2]:
        for method, n in [
            (cosmologix.interpolation.newton_interpolant, 10),
            (cosmologix.interpolation.barycentric_interpolant, 10),
            (cosmologix.interpolation.linear_interpolant, 1000),
            (jnp_interpolant, 1000),
        ]:
            print(interpolation_accuracy(func, method, n=n))


if __name__ == "__main__":
    test_interpolation()
