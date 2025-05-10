# About

## Dependencies

- JAX for numerical computations and automatic differentiation.
- NumPy for array operations (used indirectly via JAX).
- Matplotlib for plotting.
- Requests to retrieve external data files.
- tqdm to display progression of contour computation

## Roadmap

- [ ] Add 1d profile likelihoods for corner plots
- [ ] Ease the handling of exploration results

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
numerical errors. `jax_cosmo` is not presented in this comparison
because at the time of writing it does not compute the contribution of
neutrinos to the density which prevents a relevant comparison.

![Distance modulus accuracy](https://gitlab.in2p3.fr/lemaitre/cosmologix/-/raw/master/doc/mu_accuracy.svg)

## Speed test

The plot below illustrates the computation time for a vector of
distance moduli across various redshifts, plotted against the number
of redshifts. Generally, the computation time is dominated by
precomputation steps and remains largely independent of vector size,
except in the case of `astropy` and `jax_cosmo`. We differentiate
between the first call and subsequent calls, as the initial call may
involve specific overheads. For Cosmologix, this includes
JIT-compilation times, which introduce a significant delay. Efforts
are underway to optimize this aspect. Note that we did not yet manage
jit-compile the luminosity distance computation in `cosmoprimo`, due
to a compilation error. The speed measurement may change significantly
when this issue is solved.

![Distance modulus speed](https://gitlab.in2p3.fr/lemaitre/cosmologix/-/raw/master/doc/mu_speed.svg)

## Contributing

Contributions are welcome! Please fork the repository, make changes, and submit a pull request. Here are some guidelines:

- Follow PEP 8 style. The commited code has to go through black.
- Write clear commit messages.
- Include tests for new features or bug fixes.

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