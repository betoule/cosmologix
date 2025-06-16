---
title: 'Cosmologix: Fast, accurate and differentiable distances in the universe with JAX'
tags:
  - Python
  - cosmology
  - jax
  - distances
authors:
  - name: Betoule, Marc
    corresponding: true
    orcid: 0000-0003-0804-836X
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Neveu, Jérémy
    orcid: 0000-0002-6966-5946
    affiliation: 2
  - name: Kuhn, Dylan
    orcid: 0009-0005-8110-397X
    affiliation: 1
  - name: Le Jeune, Maude
    orcid: 0000-0002-1008-3394
    affiliation: 3
  - name: Bernard, Mathieu
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Regnault, Nicolas
    orcid: 0000-0001-7029-7901
    affiliation: 1
  - name: Bongard, Sébastien
    orcid: 0000-0002-3399-4588
    affiliation: 1
affiliations:
 - name: LPNHE, CNRS, France
   index: 1
   ror: 01hg8p552
 - name: IJCLab, Orsay, France
   index: 2
   ror: 03gc1p724
 - name: APC, Paris, France
   index: 3
   ror: 03tnjrr49
date: 05 February 2025
bibliography: paper.bib

---

# Summary

Type-Ia supernovae serve as standardizable candles to measure
luminosity distances in the universe. Cosmologix accelerates and
simplifies cosmological parameter inference from large datasets by
providing fully differentiable calculations of the distance-redshift
relation as a function of cosmological parameters. This is achieved
through the use of JAX [@jax2018github], a Python library providing
automatic differentiation and compilation for CPU and hardware
accelerators. `Cosmologix` incorporates the density evolution of all
relevant species, including neutrinos. It also provides common
fitting formulae for the acoustic scale so that the resulting code can
be used for fast cosmological inference from supernovae in combination
with BAO or CMB distance measurements. We checked the accuracy of our
computation against `CAMB`, `CCL` and `astropy.cosmology`. We
demonstrated that our implementation is approximately ten times faster
than existing cosmological distance computation libraries, computing
distances for 1000 redshifts in approximately 500 microseconds on a
standard laptop CPU, while maintaining an accuracy of $10^{-4}$
magnitudes in the distance modulus over the redshift range $0.01 < z <
1000$.

# Statement of need

Many software are available to compute cosmological distances
including `astropy` [@astropy], `camb` [@Challinor:2011bk], `class`
[@class1], `ccl` [@ccl]. To our knowledge only `jax-cosmo` [@jaxcosmo]
and `cosmoprimo` [@cosmoprimo] provide automatic differentiation
through the use of JAX. Unfortunately, at the time of writing, the
computation in cosmoprimo does not seem to be jitable and distance
computation in jax-cosmo is neglecting contributions to the energy
density from neutrinos and photons. The accuracy of the resulting
computation is insufficient for the need of the LEMAITRE analysis, a
compilation of type-Ia Supernovae joining the very large sample of
nearby events discovered by ZTF [@rigault:2025] to higher redshift
events from the SNLS [@astier:2006] and HSC [@yasuda:2019]. The
LEMAITRE collaboration is therefore releasing its internal code for
computing cosmological distances. The computation follows standard
methods, but our JAX implementation is optimized for speed while
maintaining sufficient accuracy.

# Computations of the homogeneous background evolution

The core library offers `jax` functions to compute the evolution of
energy density in the universe (via the `cosmologix.densities` module)
and derived quantities, such as cosmological distances (via the
`cosmologix.distances` module). Details are provided in the
documentation. As an example, we highlight the speed and accuracy of
calculating the distance modulus (the logarithm of luminosity
distance) for a large number of redshifts in the following discussion.

## Accuracy

The distance computation involves the numerical evaluation of an
integral. The resolution of the quadrature used for this evaluation is
adjustable in `cosmologix`. To assess the numerical accuracy of our
baseline computation, we compared it to the same integral evaluated at
10-fold higher resolution. The difference is displayed in
\autoref{fig:accuracy} for the baseline Planck $\Lambda$CDM model,
reported in Table 1 in [@planck2018VI]. The difference in distance
modulus between the coarse (baseline) and fine resolution computation
is smaller than $10^{-4}$ mag over the redshift range $0.01 < z <
1000$, dominated by the interpolation error.

We also compared the results of various external codes to the fine
quadrature of `cosmologix` as the reference. It demonstrates agreement
within a few $10^{-5}$ magnitudes over the same redshift
range. Residual discrepancies between libraries stem from differences
in handling the effective number of neutrino species. We adopt
`CAMB`’s convention, where all species share the same temperature,
resulting in closer alignment with its predictions. We exclude
`jax_cosmo` from this comparison because it does not account for
neutrino contributions to energy density, precluding a meaningful
comparison.

![Difference in distance modulus for the Planck best-fit
$\Lambda$CDM model with respect to the higher resolution quadrature
computation in cosmologix.\label{fig:accuracy}](mu_accuracy.pdf)

## Computation speed

The computation time for a vector of distance moduli across various
redshifts is plotted in \autoref{fig:speed} as a function of the
number of redshifts requested. We differentiate between the first call
and subsequent calls, as the initial call may involve specific
overheads. For `cosmologix`, this includes JIT-compilation times,
which introduces a significant delay. In subsequent calls,
`cosmologix` overperforms all other tested codes by a significant
margin.

In addition we also timed the computation of the jacobian matrix of
the distance modulus with respect to the 9 cosmological parameters. It
is evaluated as `jax.jacfwd(mu)`. The computation time for the Jacobian
is roughly 5 times larger than the function itself. This is faster
than finite differences, which require 10 function evaluations,
reducing computation time by approximately 50\%.

![Computation speed of the distance modulus \label{fig:speed} for
various cosmological codes. The left panel displays the measured time
for the first call which integrates pre-computation and in the case of
jax codes overhead associated with jit compilation. The right panel
displays the average time measured over 10 subsequent calls. The
measurements were obtained on an Intel(R) Core(TM) i7-1165G7 CPU
clocked at 2.80GHz, without GPU acceleration.](mu_speed.pdf)

# Differentiability and likelihood maximization

Last, the code provides a framework to efficiently build frequentist
confidence contours for cosmological parameters for all measurements
whose likelihood can be expressed as a chi-square. 
\autoref{sample_contour} provides an example 2-dimensionnal
confidence region in the plane $(\Omega_{bc}, w)$ for a flat $w$-CDM
model as probed by the Union3 supernovae compilation
[@2023arXiv231112098R]. The full computation took 3.86s on an
Intel(R) Core(TM) i7-1165G7 at 2.80GHz without GPU acceleration.

![Confidence region at 68 and 95 percent for the $w$ and $\Omega_{bc}$ parameters probed by the Union3 compilation.\label{sample_contour}](sample_contour.pdf)

# References
