# About

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
