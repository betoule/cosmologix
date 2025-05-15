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
Here's a quick example to get you started (look at
[example/features.py](https://gitlab.in2p3.fr/lemaitre/cosmologix/example/features.py)
for a more complete tour of the available features):

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

## Command line interface

For most common use cases, there is also a simple command line interface to the library. You can perform fit, contour exploration and contour plotting as follows:

```bash
# First line is optional. It activates command line completion for most common shells
cosmologix --install-completion
cosmologix fit --priors PR4 --priors DESIDR2 --cosmology FwCDM -s
cosmologix explore Omega_bc w --priors PR4 --priors DESIDR2 --cosmology FwCDM -o contours.pkl
cosmologix contour contours.pkl -s -o contour.png
```

## Dependencies

- JAX for numerical computations and automatic differentiation.
- NumPy for array operations (used indirectly via JAX).
- Matplotlib for plotting.
- Requests to retrieve external data files.
- tqdm to display progression of contour computation
- typer for the cli.

## Roadmap

- [ ] Conversion of contours to different parameterisation (e.g. `Omega_bc` to `Omega_m`)

## Contributing

Contributions are welcome! Please fork the repository, make changes, and submit a pull request. Here are some guidelines:

- Follow PEP 8 style. The commited code has to go through black.
- Write clear commit messages.
- Include tests for new features or bug fixes.

## Documentation

Detailed documentation for each function and module can be found in the source code. Autodocs is in preparation [here](https://cosmologix-7920a8.pages.in2p3.fr/).

## Release history

### v0.9.6
- 1D profile likelihoods
- Group exploration results in a single file
- Improve handling of labels in corner plots
- Change name of `Omega_m` to `Omega_bc` to lift possible confusion on neutrinos contribution accounting
- Provide high level interface compatible with the command line interface
- Limit cache size inflation

### v0.9.5 (current)
- Add DESI DR2 BAO measurements (rename DESI2024 to DESIDR1 for consistency)
- Add a Planck prior consistent with what is used in DESI DR2 (named PR4)
- Various bug fixes related to jax version
- Add minimal support for corner plots

### v0.9.4
- Add SH0ES to the list of available priors
- Compute the dark energy task force Figure of Merit (FoM) from the Fisher matrix for dark energy models
- Report χ² and fit probability in addition to best-fit parameters
- Improve the estimate of contour exploration time

### v0.9.3
- Implement a cache mechanism to mitigate pre-computation delays
- Extend the set of cosmological computation available, by adding comoving volume and lookback time
- Improvements to the command line interfacements (ability to change contour thresholds)
- Add Union3 to the set of available likelihoods

### v0.9.2
- Rewrite some of the core function to improve speed of contour exploration by about 10x
- Enable exploration of curved cosmologies (solving nan issue around Omega_k = 0)

### v0.9.1
- Add a command line interface. Makes it easy to compute bestfits, and 2D Bayesian contours for a given set of constraints
- Auto-detect under-constrained parameters

### v0.9.0
- First release with complete feature set
- Accuracy tested against CAMB and CCL
- Build-in fitter and frequentist contour exploration, taking advantage of auto-diff

### v0.1.0
- Initial release
- Core distance computation available

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


