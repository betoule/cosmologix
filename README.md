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

To install JaxCosmo, you need Python 3.7 or newer. Use pip:

```sh
pip install jaxcosmo
```

Note: Make sure you have JAX installed, along with its dependencies. If you're using GPU acceleration, ensure CUDA and cuDNN are properly set up.

## Usage
Here's a quick example to get you started:

```python
from cosmologix import mu, planck2018
import jax.numpy as jnp

# flat Λ-CDM best-fit parameters to Planck 2018 are:
print(planck2018)

# Redshift values for supernovae
z_values = jnp.linspace(0.1, 1.0, 10)

# Compute distance modulus 
distance_modulus = mu(params, z_values)

print(distance_modulus)
```

## Dependencies

- JAX for numerical computations and automatic differentiation.
- NumPy for array operations (used indirectly via JAX).
- Matplotlib for plotting (optional, for examples and testing).


## Documentation

Detailed documentation for each function and module can be found in the source code or the docs (link-to-docs-if-available).

## Contributing
Contributions are welcome! Please fork the repository, make changes, and submit a pull request. Here are some guidelines:

- Follow PEP 8 style.
- Write clear commit messages.
- Include tests for new features or bug fixes.


## License
This project is licensed under the GPLV2 License - see the LICENSE.md file for details.

## Contact

For any questions or suggestions, please open an issue.


## Acknowledgments

Thanks to the JAX team for providing such an incredible tool for numerical computation in Python.
To the cosmology and astronomy community for the valuable datasets and research that inform this package.

