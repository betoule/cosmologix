![Cosmologix Logo](doc/cosmologix_logo.png)
# Cosmologix

**Cosmologix** is a Python package for computing cosmological distances
in a Friedmann–Lemaître–Robertson–Walker (FLRW) universe using JAX for
high-performance and differentiable computations. This package is
mostly intended to fit the Hubble diagram of the LEMAITRE supernovae
compilation and as such has a slightly different (and smaller) scope
than jax-cosmo, with a focus on accurate and fast luminosity
distances. It has been tested against the CCL.

## Features

- **Cosmological Distance Calculations**: Compute various distances (comoving, luminosity, angular diameter) in an FLRW universe.
- **Hubble Diagram Fitting**: Tools to fit supernovae data to cosmological models.
- **JAX Integration**: Leverage JAX's automatic differentiation and JIT compilation for performance.
- **Neutrino Contributions**: Account for both relativistic and massive neutrinos in cosmological models.
- **CMB Prior Handling**: Includes functionality to incorporate geometric priors from CMB measurements.

## Installation


To install `cosmologix`, you need Python 3.10 or newer. Use pip:

```sh
pip install cosmologix
```

Note: Make sure you have JAX installed, along with its dependencies. If you're using GPU acceleration, ensure CUDA and cuDNN are properly set up.

## Usage
Here's a quick example to get you started:

```python
from cosmologix import mu, Planck18
import jax.numpy as jnp

# Best-fit parameters to Planck 2018 are:
print(Planck18)

# Redshift values for supernovae
z_values = jnp.linspace(0.1, 1.0, 10)

# Compute distance modulus 
distance_modulus = mu(Planck18, z_values)
print(distance_modulus)

# Find bestfit flat w-CDM cosmology
from cosmologix import likelihoods, fit
priors = [likelihoods.Planck2018Prior(), likelihoods.DES5yr()]
fixed = {'Omega_k':0., 'm_nu':0.06, 'Neff':3.046, 'Tcmb': 2.7255, 'wa':0.0}
result = fit(priors, fixed=fixed, verbose=True)
print(result['bestfit'])

# Compute frequentist confidence contours
# The progress bar provides a rough upper bound on computation time because 
# the actual size of the explored region is unknown at the start of the calculation.
# Improvements to this feature are planned.

from cosmologix import contours
grid = contours.frequentist_contour_2D_sparse(
    priors,
    grid={'Omega_m': [0.18, 0.48, 30], 'w': [-0.6, -1.5, 30]},
    fixed=fixed
    )

import matplotlib.pyplot as plt
contours.plot_contours(grid, filled=True, label='CMB+SN')
plt.ion()
plt.legend(loc='lower right', frameon=False)
plt.show()
```

Further guidance can be found reading files in the examples directory.

## Dependencies

- JAX for numerical computations and automatic differentiation.
- NumPy for array operations (used indirectly via JAX).
- Matplotlib for plotting.
- Requests to retrieve external data files.
- tqdm to display progression of contour computation

## Roadmap

- [ ] Improve speed, especially on first call as compilation time are currently a bit long (see graph below)
- [ ] Make distances differentiable around the crossing Omega_k = 0 to allow fitting non flat universe
- [ ] Add Union likelihood
- [ ] Add proper weights to the DES-5y likelihood and check resulting constraints
- [ ] Add computation of comoving volume
- [ ] Improve the guess of contour computation time

## Accuracy of the distance modulus computation

The plot below compares the distance modulus computation for the
baseline Planck 2018 flat Λ-CDM cosmological model across several
codes, using the fine quadrature of Cosmologix as the reference. It
demonstrates agreement within a few 10⁻⁵ magnitudes over a broad
redshift range. Residual discrepancies between libraries stem from
differences in handling the effective number of neutrino species. We
adopt the convention used in CAMB (assuming all species share the same
temperature), which explains the closer alignment. A comparison with
the coarse quadrature (Cosmologix 1000) highlights the magnitude of
numerical errors.

![Distance modulus accuracy](doc/mu_accuracy.svg)

## Speed test

The plot below illustrates the computation time for a vector of
distance moduli across various redshifts, plotted against the number
of redshifts. Generally, the computation time is dominated by
precomputation steps and remains largely independent of vector size,
except in the case of Astropy. We differentiate between the first call
and subsequent calls, as the initial call may involve specific
overheads. For Cosmologix, this includes JIT-compilation times, which
introduce a significant delay. Efforts are underway to optimize this
aspect.

![Distance modulus speed](doc/mu_speed.svg)

## Contributing

Contributions are welcome! Please fork the repository, make changes, and submit a pull request. Here are some guidelines:

- Follow PEP 8 style. The commited code has to go through black.
- Write clear commit messages.
- Include tests for new features or bug fixes.

## Documentation

Detailed documentation for each function and module can be found in the source code. Autodocs is in preparation [here](https://cosmologix-7920a8.pages.in2p3.fr/).



## License
This project is licensed under the GPLV2 License - see the LICENSE.md file for details.

## Contact

For any questions or suggestions, please open an issue.

## Acknowledgments

Thanks to the JAX team for providing such an incredible tool for
numerical computation in Python.  To the cosmology and astronomy
community for the valuable datasets and research that inform this
package. We are especially grateful to the contributors to the Core
Cosmology Library [CCL](https://github.com/LSSTDESC/CCL) against which
the accuracy of this code has been tested,
[astropy.cosmology](https://docs.astropy.org/en/stable/cosmology/index.html)
for its clean and inspiring interface and of course
[jax-cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo),
pioneer and much more advanced in differentiable cosmology
computations.


