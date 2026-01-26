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

To help people comparing code parametrisation, here are the parameter transformations that we used (also in `doc/accuracy_plots.py`).

### [Core Cosmological Library](https://ccl.readthedocs.io/en/latest/)

```
from cosmologix import parameters
params = get_cosmo_params('Planck18')
params = densities.derived_parameters(params)
ccl_params = {
        "Omega_c": params["Omega_c"],
        "Omega_b": params["Omega_b"],
        "Omega_k": params["Omega_k"],
        "h": params["H0"] / 100,
        "Neff": params["Neff"],
        "m_nu": [params["m_nu"], 0, 0],
        "T_CMB": params["Tcmb"],
        "T_ncdm": 0.7137658555036082,
        "n_s": 0.9652,
        "sigma8": 0.8101,
        "transfer_function": "bbks"}
```

### [CAMB](https://camb.readthedocs.io/en/latest/)

```
from cosmologix import parameters
params = get_cosmo_params('Planck18')
params = densities.derived_parameters(params)
h = params["H0"] / 100
pars = camb.set_params(
    H0=params["H0"],
    ombh2=params["Omega_b_h2"],
    omch2=params["Omega_c"] * h**2,
    mnu=params["m_nu"],
    omk=params["Omega_k"],
    tau=0.0540,
    As=jnp.exp(3.043) / 10**10,
    ns=0.9652,
    halofit_version="mead",
    lmax=3000,
)
```

### [Astropy](https://docs.astropy.org/en/stable/cosmology/index.html)

```
from cosmologix import parameters
params = get_cosmo_params('Planck18')
params = densities.derived_parameters(params)
    h = params["H0"] / 100.0
astropy_cosmo = cosmology.w0waCDM(
        H0=params["H0"],
        Om0=params["Omega_bc"],
        Ob0=params["Omega_b"],
        Ode0=params["Omega_x"],
        m_nu=[params["m_nu"], 0, 0],
        Tcmb0=params["Tcmb"],
        Neff=params["Neff"],
        w0=params["w"],
        wa=params["wa"],
    )
```

### [JaxCosmo](https://jax-cosmo.readthedocs.io/en/latest/)

```
from cosmologix import parameters
params = get_cosmo_params('Planck18')
params = densities.derived_parameters(params)
h = params["H0"] / 100
omega_b = params["Omega_b_h2"] / h**2
a = 1 / (1 + z)
jaxcosmo = jc.Cosmology(
    Omega_c=params["Omega_bc"] - omega_b,
    Omega_b=omega_b,
    h=h,
    Omega_k=params["Omega_k"],
    n_s=0.96,
    sigma8=0.8,
    w0=params["w"],
    wa=params["wa"],
)
```
    
### [CosmoPrimo](https://cosmoprimo.readthedocs.io/en/latest/)

```
from cosmologix import parameters
params = get_cosmo_params('Planck18')
params = densities.derived_parameters(params)
h = params["H0"] / 100
omega_b = params["Omega_b_h2"] / h**2
c = cosmoprimo.Cosmology(
    engine="eisenstein_hu",
    h=h,
    Omega_b=omega_b,
    Omega_cdm=params["Omega_bc"] - omega_b,
    Omega_k=params["Omega_k"],
    Tcmb=params["Tcmb"],
    w0_fld=params["w"],
    wa_fld=params["wa"],
    m_ncdm=params["m_nu"],
)
```

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
