# Release history

## Roadmap

- [ ] Conversion of contours to different parameterisation (e.g. `Omega_bc` to `Omega_m`)

### v0.9.8 (in prep.)
- API refinement:
  - introduction of `parameters.get_cosmo_params`
  - deprecation of `parameters.lcdm_deviation`
- Improved docstrings
- Improved documentation
- Clarify the scaling of the DETF FoM
- Enable storing files in ASDF format
- Shorten the paper
- Minor bug Fixes

### v0.9.7 (current)
- Improved documention
- Refined API to simplify providing distance measurements

### v0.9.6
- 1D profile likelihoods
- Group exploration results in a single file
- Improve handling of labels in corner plots
- Change name of `Omega_m` to `Omega_bc` to lift possible confusion on neutrinos contribution accounting
- Provide high level interface compatible with the command line interface
- Limit cache size inflation

## v0.9.5
- Add DESI DR2 BAO measurements (rename DESI2024 to DESIDR1 for consistency)
- Add a Planck prior consistent with what is used in DESI DR2 (named PR4)
- Various bug fixes related to jax version
- Add minimal support for corner plots

## v0.9.4
- Add SH0ES to the list of available priors
- Compute the dark energy task force Figure of Merit (FoM) from the Fisher matrix for dark energy models
- Report χ² and fit probability in addition to best-fit parameters
- Improve the estimate of contour exploration time

## v0.9.3
- Implement a cache mechanism to mitigate pre-computation delays
- Extend the set of cosmological computation available, by adding comoving volume and lookback time
- Improvements to the command line interfacements (ability to change contour thresholds)
- Add Union3 to the set of available likelihoods

## v0.9.2
- Rewrite some of the core function to improve speed of contour exploration by about 10x
- Enable exploration of curved cosmologies (solving nan issue around Omega_k = 0)

## v0.9.1
- Add a command line interface. Makes it easy to compute bestfits, and 2D Bayesian contours for a given set of constraints
- Auto-detect under-constrained parameters

## v0.9.0
- First release with complete feature set
- Accuracy tested against CAMB and CCL
- Build-in fitter and frequentist contour exploration, taking advantage of auto-diff

## v0.1.0
- Initial release
- Core distance computation available
