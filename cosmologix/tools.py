import jax.numpy as jnp
from jax import lax
import jax
from typing import Callable, Tuple
import requests
import numpy as np

def restrict(f: Callable, fixed_params: dict = {}) -> Callable:
    """
    Modify a function by fixing some of its parameters.

    This is similar to functools.partial but allows fixing parts of the first pytree argument.

    Parameters:
    -----------
    f: Callable
        A function with signature f(params, *args, **keys) where params is a pytree.
    fixed_params: dict
        Parameters to fix with provided values.

    Returns:
    --------
    Callable
        Function with same signature but with parameters fixed to their provided values.

    Example:
    --------
    If mu expects a dictionary with 'Omega_m' and 'w',
    restrict(mu, {'w': -1}) returns a function of 'Omega_m' only.
    """

    def g(params, *args, **kwargs):
        updated_params = fixed_params.copy()
        updated_params.update(params)
        return f(updated_params, *args, **kwargs)

    return g


def safe_vmap(in_axes: Tuple[None | int, ...] = (None, 0)) -> Callable:
    """
    Vectorize a function with JAX's vmap, treating all inputs as arrays.

    Converts scalar inputs to 1D arrays before applying vmap, ensuring 1D array outputs.

    Parameters:
    - in_axes (Tuple[None | int, ...]): Specifies which dimensions of inputs to vectorize.

    Returns:
    - Callable: JIT-compiled, vectorized version of the input function.
    """

    def wrapper(f: Callable) -> Callable:
        @jax.jit
        def vectorized(*args):
            vargs = [
                jnp.atleast_1d(arg) if ax is not None else arg
                for arg, ax in zip(args, in_axes)
            ]
            result = jax.vmap(f, in_axes=in_axes)(*vargs)
            return result

        return vectorized

    return wrapper


def trapezoidal_rule_integration(
    f: Callable, bound_inf: float, bound_sup: float, n_step: int = 1000, *args, **kwargs
) -> float:
    """
    Compute the integral of f over [bound_inf, bound_sup] using the trapezoidal rule.

    Parameters:
    -----------
    f: Callable
        Function to integrate.
    bound_inf, bound_sup: float
        Integration bounds.
    n_step: int
        Number of subdivisions.
    *args, **kwargs:
        Additional arguments passed to f.

    Returns:
    --------
    float: The computed integral.
    """
    x = jnp.linspace(bound_inf, bound_sup, n_step)
    y = f(x, *args, **kwargs)
    h = (bound_sup - bound_inf) / (n_step - 1)
    return (h / 2) * (y[1:] + y[:-1]).sum()


def linear_interpolation(
    x: jnp.ndarray, y_bins: jnp.ndarray, x_bins: jnp.ndarray
) -> jnp.ndarray:
    """
    Perform linear interpolation between set points.

    Parameters:
    -----------
    x: jnp.ndarray
        x coordinates for interpolation.
    y_bins, x_bins: jnp.ndarray
        y and x coordinates of the set points.

    Returns:
    --------
    jnp.ndarray: Interpolated y values.
    """
    bin_index = jnp.digitize(x, x_bins) - 1
    w = (x - x_bins[bin_index]) / (x_bins[bin_index + 1] - x_bins[bin_index])
    return (1 - w) * y_bins[bin_index] + w * y_bins[bin_index + 1]


class Constants:
    G = 6.67384e-11  # m^3/kg/s^2
    c = 299792458.0  # m/s
    pc = 3.08567758e16  # m
    mp = 1.67262158e-27  # kg
    h = 6.62617e-34  # J.s
    k = 1.38066e-23  # J/K
    e = 1.60217663e-19  # C
    sigma = 2 * jnp.pi**5 * k**4 / (15 * h**3 * c**2)  # Stefan-Boltzmann constant
    qmax = 30
    nq = 100
    const = 7.0 / 120 * jnp.pi**4
    const2 = 5.0 / 7.0 / jnp.pi**2
    N = 2000
    am_min = 0.01
    am_max = 600.0
    zeta3 = 1.2020569031595942853997
    zeta5 = 1.0369277551433699263313
    neutrino_mass_fac = (
        94.082  # Conversion factor for thermal neutrinos with Neff=3, TCMB=2.7255
    )
    
def load_csv_from_url(url, delimiter=','):
    """
    Load a CSV file from a URL directly into a NumPy array without saving to disk.
    
    Parameters:
    - url (str): URL of the CSV file to download.
    - delimiter (str): The string used to separate values in the CSV file (default is ',').
    
    Returns:
    - numpy.ndarray: The loaded CSV data as a NumPy array.
    """
    response = requests.get(url)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    
    # Decode the response content and split into lines
    lines = response.content.decode('utf-8').splitlines()
    
    # Process the CSV data
    def convert(value):
        ''' Attemps to convert values to numerical types
        '''
        value = value.strip()
        if not value:
            value = "nan"
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        return value
    header = lines[0].split(delimiter)
    data = [[convert(value) for value in line.split(delimiter)] for line in lines[1:]]
        
    return np.rec.fromrecords(data, names=header)
