# Accuracy and speed

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

The JAX implementation also enables seamless utilization of hardware
accelerators, such as GPUs. However, the CPU-based computation is
already highly efficient. To maintain high accuracy in distance
calculations, double-precision floating-point arithmetic is currently
required, which may necessitate adjustments to fully leverage GPU
performance benefits. Given limited motivation to pursue further
optimization, we conducted only minimal GPU testing, which indicated
that the code, in its present form, does not gain significant
performance advantages from GPU execution.
