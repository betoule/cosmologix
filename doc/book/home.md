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
