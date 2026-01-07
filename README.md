![Cosmologix Logo](https://gitlab.in2p3.fr/lemaitre/cosmologix/-/raw/master/doc/cosmologix_logo.png)
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
- **JAX Integration**: Leverage JAX's automatic differentiation and JIT compilation for performance.
- **Neutrino Contributions**: Account for both relativistic and massive neutrinos in cosmological models.
- **CMB Prior Handling**: Includes geometric priors from CMB and BAO measurements.

![Features](https://gitlab.in2p3.fr/lemaitre/cosmologix/-/raw/master/doc/features.svg)

## Installation


To install `cosmologix`, you need Python 3.10 or newer. Use pip:

```sh
pip install cosmologix
```

Note: Make sure you have JAX installed, along with its dependencies. If you're using GPU acceleration, ensure CUDA and cuDNN are properly set up.

## Usage

### Command line interface

For most common use cases, there is a simple command line interface to the library. You can perform fit, contour exploration and contour plotting as follows:

```bash
# First line is optional. It activates command line completion for most common shells
cosmologix --install-completion
cosmologix fit --priors PR4 --priors DESIDR2 --cosmology FwCDM -s
cosmologix explore Omega_bc w --priors PR4 --priors DESIDR2 --cosmology FwCDM -o contours.pkl
cosmologix contour contours.pkl -s -o contour.png
```

### Lower level interface

For other use cases, here's a quick example to get you started with
the internals of the library. Look at the
[tutorial](https://lemaitre.pages.in2p3.fr/cosmologix/basic.html) for
a more complete tour of the available features.

```python
from cosmologix import distances, parameters
import jax.numpy as jnp

# Best-fit parameters to Planck 2018 are:
print(parameters.Planck18)

# Redshift values for supernovae
z_values = jnp.linspace(0.1, 1.0, 10)

# Compute distance modulus 
distance_modulus = distances.mu(parameters.Planck18, z_values)
print(distance_modulus)

# Find bestfit flat w-CDM cosmology
from cosmologix import likelihoods, fitter
# At first call the following will download and cache the relevant data (internet connexion required)
priors = [likelihoods.Planck2018(), likelihoods.DES5yr()]
fixed = {'Omega_k':0., 'm_nu':0.06, 'Neff':3.046, 'Tcmb': 2.7255, 'wa':0.0}

result = fitter.fit(priors, fixed=fixed, verbose=True)
print(result['bestfit'])

# Compute frequentist confidence contours
# The progress bar provides a rough upper bound on computation time because 
# the actual size of the explored region is unknown at the start of the calculation.
# Improvements to this feature are planned.

from cosmologix import contours, display
grid = contours.frequentist_contour_2d_sparse(
    priors,
    grid={'Omega_bc': [0.18, 0.48, 30], 'w': [-0.6, -1.5, 30]},
    fixed=fixed
    )

import matplotlib.pyplot as plt
display.plot_contours(grid, filled=True, label='CMB+SN')
plt.ion()
plt.legend(loc='lower right', frameon=False)
plt.show()
#Further examples can be found reading files in the examples directory, especially example/features.py.
```

## Documentation

The complete documentation is available
[here](https://lemaitre.pages.in2p3.fr/cosmologix/home.html). It
includes a
[tutorial](https://lemaitre.pages.in2p3.fr/cosmologix/basic.html),
and full [API](https://lemaitre.pages.in2p3.fr/cosmologix/autoapi/cosmologix/index.html) documentation.

## Contributing

Contributions are welcome! Please fork the repository, make changes, and submit a pull request. Here are some guidelines:

- Follow PEP 8 style. The submitted code has to go through black.
- Write clear commit messages.
- Include tests for new features or bug fixes.
- Run the test suite before submission

Running the test suite requires a few more package:
```bash
git clone https://gitlab.in2p3.fr/lemaitre/cosmologix.git
pip install .[test]
pytest -v
```
```
```

Look at the [Road map and release history
page](https://lemaitre.pages.in2p3.fr/cosmologix/release.html) for
ongoing and planned implementation of new features.


## Dependencies

- JAX for numerical computations and automatic differentiation.
- NumPy for array operations (used indirectly via JAX).
- Matplotlib for plotting.
- Requests to retrieve external data files.
- tqdm to display progression of contour computation
- typer for the cli.
- astropy for reading fits tables.
- asdf results can be written in the ASDF format.
- zstandard for file compression.

A few optional dependencies are necessary to run the test suite and some of the provided examples, or useful for the development:

- pytest
- pytest-cov for coverage reports
- pyccl for accuracy tests
- pyyaml
- black for code formating
- scipy for accuracy tests
- camb for accuracy tests
- jax_cosmo for accuracy and performance tests

Install with `pip install cosmologix[test]` to retrieve the optional dependencies.

## License
This project is licensed under the GPLV2 License - see the LICENSE.md file for details.

## Contact

For any questions or suggestions, please open an [issue](https://gitlab.in2p3.fr/lemaitre/cosmologix/-/issues).

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


