# More advanced topics

## Fixing Unconstrained Parameters

Cosmologix uses a default set of cosmological parameters in its
computations: `{'Tcmb', 'Omega_bc', 'H0', 'Omega_b_h2', 'Omega_k', 'w',
'wa', 'm_nu', 'Neff'}`. However, certain combinations of cosmological
probes may be entirely insensitive to some of these parameters,
requiring their values to be fixed for the fitting process to
converge. For instance, the cosmic microwave background temperature
(`Tcmb`) is usually assumed constant in many analyses. Late-time
probes of the expansion history—like supernovae or uncalibrated baryon
acoustic oscillations (BAOs)—do not distinguish between baryon and
dark matter contributions (`Omega_b_h2`) or constrain the absolute
distance scale (`H0`), leaving these parameters effectively
unconstrained without additional data.

### Setting Fixed Parameters
In Cosmologix, you can fix parameters by passing the optional `fixed`
argument to the `fit` and `contours.frequentist_contour_2d_sparse`
functions. This mechanism also enables exploration of simplified
cosmological models, such as enforcing flatness (`Omega_k = 0`) or a
cosmological constant dark energy behavior (`w = -1`, `wa = 0`):

```python
fixed = {'Omega_k': 0.0, 'm_nu': 0.06, 'Neff': 3.046, 'Tcmb': 2.7255, 'wa': 0.0}
result = fit(priors, fixed=fixed)
grid = contours.frequentist_contour_2d_sparse(
    priors,
    grid={'Omega_bc': [0.18, 0.48, 30], 'w': [-0.6, -1.5, 30]},
    fixed=fixed
)
```

### Degeneracy Checks
Recent versions of Cosmologix include a safeguard in the `fit`
function: it checks for perfect degeneracies among the provided priors
and fixed parameters before proceeding, raising an explicit error
message if any remain. The `contours.frequentist_contour_2d_sparse`
function, however, skips this check to allow exploration of partially
degenerate parameter combinations, offering flexibility for diagnostic
purposes.

### Command-Line Interface
From the command line, you can specify fixed parameters using the `-F`
or `--fixed` option, available for both `fit` and `explore`
commands. Additionally, the `-c` or `--cosmo` shortcut simplifies
restricting the model to predefined configurations (e.g., flat \( w
\)CDM):

```bash
cosmologix fit -p DESIDR2 -F H0 70 -c FwCDM
cosmologix explore Omega_bc w -p DESIDR2 -c FwCDM -F H0 70 -o desi_fwcdm.pkl
```

### Automatic Parameter Fixing
For convenience, the `fit` command offers the `-A` or
`--auto-constrain` option, which automatically identifies and fixes
poorly constrained parameters. Use this with caution, as it may alter
the model by trimming parameters that lack sufficient constraints,
potentially affecting your results:

```bash
cosmologix fit -p DES5yr -A -c FwCDM
```

Example output:
```
Unconstrained Parameters:
  Omega_b_h2: FIM = 0.00 (effectively unconstrained)
Fixing unconstrained parameter Omega_b_h2
Try again fixing H0
Omega_bc = 0.272 ± 0.089
w = -0.82 ± 0.17
M = -0.053 ± 0.013
```

## Cache Mechanism

Cosmologix includes a caching system to optimize performance by storing results from time-consuming operations. This mechanism applies to:
- Downloading external files, such as datasets.
- Expensive computations, like matrix inversions or factorizations used in \( \chi^2 \) calculations.
- Lengthy `jax.jit` compilations, which can have noticeable pre-run delays.

Caching helps reduce the initial overhead (sometimes called "preburn time") introduced by JAX’s just-in-time compilation and other resource-intensive tasks, making subsequent runs significantly faster.

### Accessing the Cache Directory
You can retrieve the location of the cache directory using the `tools` module:

```python
from cosmologix import tools
print(tools.get_cache_dir())
```

This returns the path where cached files are stored, typically a platform-specific directory (e.g., `~/.cache/cosmologix` on Unix-like systems).

### Managing the Cache
If the cache grows too large or if you suspect outdated results are being loaded due to code changes, you can clear it entirely:

```python
tools.clear_cache()
```

This removes all cached files, forcing Cosmologix to recompute or redownload as needed on the next run. You can delete only the jit-compilation cache, avoiding the need to redownload all data with:
```python
tools.clear_cache(jit=True)
```

You can also perform the same operations from the command line:
```bash
cosmologix clear-cache
cosmologix clear-cache -j
```

### Notes

- The caching system is particularly useful for mitigating JAX’s compilation delays, but its effectiveness depends on consistent inputs and code stability.
- Use `clear_cache()` judiciously, as it deletes all cached data, including potentially large datasets, and will require internet connexion to download.
- Cache inflation is generally caused by the accumulation of compiled jit code for different combination of likelihoods and cosmologies. To avoid disk space issues the use of the persistent cache for jit code is deactivated when the cache size exceeds 1GB. 
