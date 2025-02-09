import jax.numpy as jnp
from jax import lax
import jax
from typing import Callable, Tuple
import requests
import numpy as np
import os
import hashlib
from pathlib import Path
import shutil


def get_cache_dir():
    """Determine the appropriate cache directory based on the OS."""
    if os.name == "nt":  # Windows
        return str(Path(os.getenv("LOCALAPPDATA")) / "Cache" / "cosmologix")
    elif os.name == "posix":  # Unix-like systems
        if "XDG_CACHE_HOME" in os.environ:
            return str(Path(os.environ["XDG_CACHE_HOME"]) / "cosmologix")
        return str(Path.home() / ".cache" / "cosmologix")
    else:
        raise OSError("Unsupported operating system")


def clear_cache(cache_dir=None):
    """
    Clear the cache directory used by cached_download.

    :param cache_dir: Optional directory path for cache. If None, it uses an OS-specific default.
    :return: None
    """
    if cache_dir is None:
        cache_dir = (
            get_cache_dir()
        )  # Assuming get_cache_dir() is defined as in the previous example

    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)  # Remove the entire directory
            print(f"Cache directory {cache_dir} has been cleared.")
        except Exception as e:
            print(f"An error occurred while clearing the cache: {e}")
    else:
        print(f"Cache directory {cache_dir} does not exist.")


def cached_download(url, cache_dir=None):
    """
    Download a file from the web with caching.

    :param url: The URL to download from.
    :param cache_dir: Optional directory path for cache. If None, it uses an OS-specific default.
    :return: Path to the cached file.
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Use URL hash for filename to avoid conflicts
    filename = hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path):
        print(f"Using cached file: {cache_path}")
        return cache_path

    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(cache_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"File downloaded and cached at: {cache_path}")
    return cache_path


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


def load_csv_from_url(url, delimiter=","):
    """
    Load a CSV file from a URL directly into a NumPy array

    Parameters:
    - url (str): URL of the CSV file to download.
    - delimiter (str): The string used to separate values in the CSV file (default is ',').

    Returns:
    - numpy.ndarray: The loaded CSV data as a NumPy array.
    """
    # response = requests.get(url)
    # response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

    # Decode the response content and split into lines
    # lines = response.content.decode("utf-8").splitlines()
    path = cached_download(url)
    with open(path, "r") as fid:
        lines = fid.readlines()

    # Process the CSV data
    def convert(value):
        """Attemps to convert values to numerical types"""
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


def conflevel_to_delta_chi2(level, dof=2, max_iter=1000, tol=1e-6):
    '''Return the ΔΧ² value corresponding to a given
    confidence level in percent

    parameter:
    - level: confidence level in percent
    - dof: Number of degrees of freedom of the chi2 law
    - max_iter: maximum number of iteration in the Newton search.
    - tol: The tolerance for the numerical accuracy of the solution.
    Note: for some reason ppf is not available in jax.scipy but cdf
    is, a quick fix was to solve for the root using Newton method.
    '''
    x = jnp.array(dof)
    prob= level/100
    def f(x):
        return jax.scipy.stats.chi2.cdf(x, dof) - prob

    def df(x):
        return jax.scipy.stats.chi2.pdf(x, dof)
    
    for _ in range(max_iter):
        x_new = x - f(x) / df(x)
        if jnp.abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise ValueError("Newton's method did not converge")

# This global random key at the module level is provided for
# conveniency so that random vector can be obtained with onliners when
# need. This will note ensure actual randomness nor reproducibility.
# To be used cautiously
global_key = jax.random.PRNGKey(42)

def randn(sigma, n=None, key=None):
    """
    Generate a Gaussian random vector scaled by sigma.

    Parameters:
    -----------
    sigma : float or array_like
        Standard deviation to scale the random vector. If sigma is an array, 
        each element scales the corresponding element of the output vector.
    n : int or tuple, optional
        If provided, specifies the shape of the output. If not provided (None),
        the shape of `sigma` is used.
    key : jax.random.PRNGKey, optional
        PRNG key for random number generation. If not provided, uses the 
        global key, which can lead to non-reproducible results.

    Returns:
    --------
    ndarray
        An array of Gaussian random numbers scaled by `sigma`.

    Notes:
    ------
    - Using the global key (`key=None`) can lead to correlated results across 
      function calls since the key is updated globally. For reproducibility, 
      pass an explicit key.
    - The function splits the key if no key is provided, which might lead 
      to issues in parallel computing scenarios due to key reuse.

    Examples:
    ---------
    >>> import jax.numpy as jnp
    >>> randn(1.0, (2, 3))  # 2x3 matrix of random numbers scaled by 1.0
    >>> randn(jnp.array([1, 2, 3]))  # Vector scaled by different sigmas
    """ 
    global global_key
    if key is None:
        global_key, subkey = jax.random.split(global_key)
    else:
        subkey = key
    if n is None:
        n = sigma.shape
    gaussian_vector = jax.random.normal(subkey, n)
    return gaussian_vector * sigma
