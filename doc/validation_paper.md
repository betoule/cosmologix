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
computing cosmological distances. While the computation follows
standard methods, our JAX implementation is optimized for speed while
maintaining sufficient accuracy. This paper documents the
implementation. Readers unfamiliar with cosmological computations may
find helpful guidance in standard cosmology resources, such as
Baumann's course [@baumann].

# Computations of the homogeneous background evolution

The core of the library provides `jax` functions to compute the
evolution of energy density in the universe (module
`cosmologix.densities`) and use them to provide efficient computation
of derived quantities such as cosmological distances (module
`cosmologix.distances`). The goal of this section is to specifically
document the core of the implementation. We adopt commonly used
notations and the code itself follow the same notations, spelling-out
greek letters, whenever possible.

## Friedmann equations

All computations in `cosmologix` are made for the
Friedman-Lemaitre-Robertson-Walker metric (isotropic and homogeneous
universe), whose length element $ds$ writes as a function of cosmological
time $t$, scale factor $R(t)$ and spherical spatial coordinates $(r, \theta, \phi)$:
\begin{equation}
  \label{eq:27}
  ds^2 = -c^2dt^2 + R^2(t) \left(\frac{dr^2}{1-kr^2} + r^2(d\theta^2 +
    \sin^2\theta d\phi^2) \right) \quad \text{with} \quad k = \lbrace-1, 0, 1\rbrace.
\end{equation}
Denoting $a(t) = \frac{R(t)}{R_0}$ and:
$$
S=\left\lbrace
    \begin{array}{l}
      \sinh \text{ if }k = 1\\
      \mathrm{Id} \text{ if }k = 0\\
      \sin \text{ if }k = -1\\
    \end{array}
  \right.,
$$
we define the comoving coordinate $\chi$ as $r = S\left(\frac{\chi}{R_0}\right)$. The metrics then rewrites:
\begin{equation}
    ds^2 = -c^2 dt^2 +  a^2(t)\left( d\chi^2 + R_0^2 S^2\left(\frac{\chi}{R_0}\right) d\Omega^2\right).
\end{equation}

The first Friedman equation without the cosmological
constant term (whose role will be held by the fluid) reads:

\begin{equation}
  \label{eq:2}
  H^2 = \frac{8\pi G}{3}\rho - \frac{k}{R^2}\,,
\end{equation}

where $\rho$ is the proper energy density, $H =
\dot R / R$ the Hubble parameter and $k = {-1, 0, 1}$ is the sign of
spatial curvature. The value of constants (such as $G$) used in the
code are given in \autoref{tab:constants}.

: Physical constants used in the code\label{tab:constants}.

| Variable | name                   | Value          | Unit                  |
|----------|------------------------|----------------|-----------------------|
| $G$      | Gravitational constant | 6.67384e-11    | m$^3$kg$^{-1}s$^{-2}$ |
| $c$      | Speed of light         | 299792458.0    | m/s                   |
| pc       | Parsec                 | 3.08567758e16  | m                     |
| $m_p$    | Proton mass            | 1.67262158e-27 | kg                    |
| $h$      | Planck constant        | 6.62617e-34    | J.s                   |
| $k$      | Boltzman constant      | 1.38066e-23    | J/K                   |
| $e$      | Electron charge        | 1.60217663e-19 | C                     |



Denoting, as usual, $\rho_{c} = \frac{3 H_0^2}{8\pi G}$ the
critical value of the density for which the universe today is flat,
$\Omega_0 = \frac{\rho_0}{\rho_{c}}$ the reduced energy density
today and $\Omega_k = -\frac{k}{R^2_0H^2_0}$, one can rewrite the equation
under its most common form: \begin{equation} \label{eq:3}
\frac{H^2}{H_0^2} = \Omega_0 \frac{\rho}{\rho_0} + \Omega_k (1+z)^2,
\end{equation}
with the redshift $z$ defined as $1+z = 1/a$. 

## Densities

We follow the standard practice of dividing the universe's energy
content into components such as cold (pressureless) matter, radiation,
and dark energy.  Most of the components are parameterized in the code
by their reduced density today and the parameter of their equation of
state. The exceptions are photons, for which we pass the observed
temperature of the CMB $T_{cmb}$ instead, and for neutrinos which, in
the general case, transition from ultra-relativistic to
non-relativistic over the period of interest and whose energy density
evolution requires a specific numerical computation.

### Species parameterized by reduced density and equations of state

For a perfect fluid $x$ with equation of state $\rho_x = w_x p_x$, the
energy conservation writes:

\begin{equation}
  \label{eq:4}
  \frac{d}{dt}(\rho_x R^3) = -3 p_x R^2 \dot R = -3 w_x \rho_x R^2 \dot R\,,
\end{equation}
which can be integrated to give:

\begin{equation}
  \label{eq:5}
  \log \frac{\rho_x}{\rho_x^0} = 3 \int_0^z \frac{1+w_x(z)}{1+z}dz\,.
\end{equation}

In the code the following components follow this description, with
simplification made when appropriate:

- Baryonic matter: $\Omega_b, w_b=0$  which gives $\frac{\rho_b}{\rho_b^0} = (1+z)^3$,
- Cold dark matter: $\Omega_c, w_c=0$ which gives $\frac{\rho_c}{\rho_c^0} = (1+z)^3$,
- Dark energy $\Omega_x$.
We allow dark energy to have a variable equation of state according to the common CPL [@cheval:2001] parametrization: $w(z) = w + \frac{z}{1+z} w_a$ with $w$ and $w_a$ as free parameters. Once integrated this gives the following evolution for the contribution to density:

$$\rho_x/\rho_0 = \exp\left(3 (1 + w + w_a) \log(1+z) - 3 w_a \frac{z}{1+z}\right).$$
  
  
### Relic photons

In the more general case, the energy density of species $i$ is obtained
by integration over the distribution function: 
\begin{equation}
\label{eq:12} \rho_i = g_i \int n_i(p) E(p) \frac{4\pi p^2dp}{h^3}
\end{equation} where $g_i$ is the degeneracy number of the species.

For photons with two spin states this will reduce to Stephan's law. Given that:
\begin{equation}
  \label{eq:14}
  n_\gamma(p) = \frac{1}{e^{\frac{cp}{k_B T_\gamma}} + 1}, \quad g_i=2,
\end{equation}
we obtain, with the variable change $x=\frac{cp}{k_BT_\gamma}$:
\begin{equation}
  \label{eq:15}
  \rho_\gamma = \frac{8\pi(k_B T_\gamma)^4}{ c^3 h^3}  \int_0^\infty \frac{x^3}{e^x - 1} dx=\frac{8\pi^5(k_B T_\gamma)^4}{15 c^3 h^3} 
\end{equation}
where we have used the result $\int_0^\infty u^{s-1}/(e^u - 1) du =
\Gamma(s) \zeta(s)$. 

Instead of providing $\Omega_\gamma^0$ the code expects the
temperature of the frozen thermal spectrum today denoted $T^0_\gamma =
T_\text{cmb}$ from which it computes $\Omega_\gamma(T_\text{cmb}, H_0) =
\rho_\gamma(T_{cmb}) / \rho_c(H_0)$. The density then evolves as
$T_\gamma^4 \propto (1+z)^4$.  As a default value the code uses
$T_\gamma^0 = 2.7255 K$ [@fixen:2009ApJ].

### Neutrinos

For neutrinos following the Fermi-Dirac distribution, neglecting the
chemical potential at high temperature, the particle density for a
neutrino species $i$ of mass $m_i$ at temperature $T_i$ is:
\begin{equation}
  \label{eq:11}
  n_i(p) = \frac{1}{e^{\frac{E}{k_B T_i}} + 1}
\end{equation}
with $E(p)^2 = c^2p^2 + m_i^2c^4$. While relativistic, with the same
variable change as above, we obtain the energy density:
\begin{equation}
  \label{eq:13}
  \rho^\text{nomass}_i = g_i \frac{4\pi(k_B T_i)^4}{ c^3 h^3}  \int_0^\infty \frac{x^3}{e^x + 1} dx = \frac{g_i 4\pi^5(k_B T_i)^4}{15 c^3 h^3}\frac{7}{8}
\end{equation}
where we have used $\int_0^\infty u^{s-1}/(e^u + 1) du = \Gamma(s)
\zeta(s) (1-1/2^{s-1})$. The universe is heated by the
electron-positron annihilation when the decoupling of neutrinos is
nearly complete. The neutrinos and photons temperatures after
annihilation are related as follows:
\begin{equation}
  \label{eq:16}
  \frac{T_\nu}{T_\gamma} = \left( \frac{4}{11}\right)^{1/3} \left(\frac{N_\text{eff}}3\right)^{1/4}
\end{equation}
with a default value for $N_\text{eff} = 3.046$ [@Mangano:2005NuPhB]. The
effective density for 6 species of relativistic neutrinos and
anti-neutrinos is:
\begin{equation}
  \label{eq:18}
  \rho^\text{nomass}_\nu = \frac78 N_\text{eff}\left(\frac{4}{11}\right)^{4/3}\rho_\gamma\,.
\end{equation}

The distribution of neutrinos after their decoupling is frozen. The energy
density for massive neutrinos, assuming that the decoupling occurs
when neutrinos are still ultra-relativistic, is given by:
\begin{align}
  \label{eq:17}
  \rho_i & = g_i \frac{4\pi(k_B T_i)^4}{ c^3 h^3}  \int_0^\infty \frac{x^3\sqrt{1 + (\bar m/x)^2}}{e^x + 1} dx\\
  & = \rho^\text{nomass}_i \frac{120}{7\pi^4} I(\bar m)\,,
\end{align}
where we denote $\bar m = m_i c^2 / (k_B T_i)$. Fast evalution of this integral is obtained as follows:
\begin{itemize}
\item ultra-relativistic case ($\bar m \leq 0.01$): the integral is evaluated analytically through the expansion $I(\bar m) \sim I(0) (1 + \frac5{7\pi^2} \bar m^2)$.
\item intermediate case $0.01 < \bar m < 1000$: $I(\bar m)$ is evaluated
  numerically at 35 Chebyshev nodes in $\log_{10}(\bar m)$. The $k^{th}$ of $n=35$ Chebyshev nodes are defined on the segment $[-1, 1]$ as $\cos(k \pi / n)$, and mapped to the segment $[-2, 3]$ in the log space. The function is then evaluated as $I(\bar m) = N(\bar m)$ where $N$ is the Newton interpolation polynomial accross the 35 precomputed nodes. The numerical pre-computation of the integral at the 35 nodes uses the trapezoidal rule over $10^4$ points in $x$ with infrared and ultraviolet cutoff at $x=10^{-3}$ and $x=31$.
\item non-relativistic case ($\bar m \geq 1000$): the integral is again evaluated analytically through the expansion $I(\bar m) \sim \frac32 \zeta(3) \bar m + \frac{3}{4\bar m} 15 \zeta(5)$ .
\end{itemize}

The relative difference between the numerical computation and the fast and composite interpolation-expansion is shown in \autoref{fig:densityinterpolation}. The approximation is shown to have a worst case accuracy better than $10^{-7}$ which is largely sufficient for this subdominant species.

![Comparison between the relatively slow numerical evaluation of integral $I(\bar m)$ and its fast interpolant. The top pannel shows the two evaluation of the function and the relative difference between the two is displayed in the bottom pannel. Vertical dotted lines display the switch between the analytical expansions and the Newton interpolant.\label{fig:densityinterpolation}](density_interpolation.pdf)

Our parametrisation of the density of neutrinos follows the common current practice to provide the value of the effective number $N_\text{eff}$ and the sum of neutrinos masses as $m_\nu$ ($0.06$ eV by default). For the computation, the entire mass is affected to one massive specie and the two others are kept massless. The code itself performs the actual computation for the three species so that this convention can be easily changed.

### Parameterization summary
To summarize, the energy content and age of the Universe are parameterized in the code using the following minimal set of parameters, given along their default value as a python dictionnary:

```python
params = {
	'Tcmb': 2.7255, # CMB temperature today in K
	'Omega_bc': 0.31315017687047186, # \Omega_b + \Omega_c
	'H0': 67.37, # Hubble constant in km/s/Mpc
	'Omega_b_h2': 0.02233, # \Omega_b * (H0/100)^2
	'Omega_k': 0.0, # \Omega_k
	'w': -1.0,
	'wa': 0.0,
	'm_nu': 0.06, # Sum of neutrinos masses in eV
	'Neff': 3.046 # Effective number of neutrino species
 }
```

We prefer the use of $\Omega_{bc}$ to $\Omega_m$ as a primary variable, because $\Omega_m$ usually incorporates the density contribution of non-relativistic neutrinos today. This computation would cause branching in the Jax code between massive and massless cases which would be detrimental to performances. The value of $\Omega_m$ and the separated density contribution of massive and mass-less neutrinos can be computed from the primary parameters using the function `densities.derived_parameters`.

## Distances

\label{sec:distances}
In the code, we denote the Hubble distance:
\begin{equation}
  \label{eq:28}
  d_H = \frac{c}{H(z)}\,.
\end{equation}
The comoving distance (coordinate) is:
\begin{equation}
  \label{eq:25}
  d_C(z) = \chi(z) = d_{H_0} \int_o^z \frac{dz'}{H(z')/H_0}\,.
\end{equation}
The integral is much easier to evaluate numerically with the following change of variable $u = (1+z)^{-1/2}$:
\begin{equation}
  \label{eq:6}
  d_C(u) =  2d_{H_0}\int_u^1 \frac{du'}{u'^3H(u')/H_0}\,.
\end{equation}

To speed up computation for large number of redshifts, the integrand
is evaluated on a fixed grid of $1000$ points regularly spanning the
interval $u=[0.02, 1]$. This corresponds to a maximum redshift
$z=2500$. The cumulative sum of the resulting vector multiplied by
the grid step gives the rectangular rule approximation of the integral
at each point in the grid in $u$. Linear interpolation in the result
is then used to reconstruct the distance values at the requested
redshifts efficiently. The quadrature resolution has been chosen to
provide interpolation errors below $10^{-4}$ over the entire redshift
range of interest. The numerical accuracy and speed of this procedure
is further assessed in \autoref{sec:numerical_results} numerical result.

The same quadrature is used to compute the look-back time:
\begin{equation}
  \label{eq:25}
  t_\text{back}(z) = \frac{1}{H_0} \int_o^z \frac{dz'}{(1+z')H/H_0}\,.
\end{equation}

The transverse comoving distance is obtained as:
\begin{equation}
  \label{eq:1}
  d_M(z) = d_{H_0} \vert\Omega_k\vert^{-1/2} S\left(\vert\Omega_k\vert^{1/2} \frac{\chi}{d_{H_0}}\right)\,.
\end{equation}
using the branch mechanism provided by `lax.switch` to deal with the cases $k=\lbrace-1, 0, 1\rbrace$ while preserving jitability and automatic differentiability.

From there the physical transverse distance, luminosity distance and distance modulus are readily obtained as:
\begin{equation}
d_A(z) = \frac{d_M(z)}{1+z}, \quad d_L(z) = d_M(z)(1+z), \quad \mu(z) = 5 \log_{10}(d_L(z)) + 25
\end{equation}

## Volume
\label{sec:volume}

The elementary comoving volume defined in a comoving distance slice $d\chi$ and solid angle $d\Omega$ is given by:
\begin{equation}
  \label{eq:24}
  dV = d\chi d_M(z)^2 d\Omega
\end{equation}
Integration in a flat universe is simpler and gives:
\begin{equation}
  \label{eq:26}
  V(z) = d\Omega \int_0^{\chi(z)}\chi^2 d\chi = \frac{\chi(z)^3d\Omega }{3}
\end{equation}
In non-flat universes, a bit of trigonometric manipulation gives:
\begin{align}
  \label{eq:26}
  V(z) &= \frac{d\Omega d_{H_0}^2}{\vert \Omega_K\vert}\int_0^{\chi(z)} S^2\left(\sqrt{\vert \Omega_k\vert}\frac{\chi}{d_{H_0}}\right) d\chi\\
  &= \frac{d\Omega d_{H_0}^2}{2 \Omega_k} \left[d_M(z) \sqrt{1 + \Omega_k \left(\frac{d_M(z)}{d_{H_0}}\right)^2} - \chi(z)\right]
\end{align}

## Sound horizon and fit formulae
\label{sec:sound-horizon}

Distance measurement calibrated on the acoustic scale requires computation of the sound horizon at a given redshift:
\begin{equation}
  \label{eq:8}
  r_s(z) = \frac{c}{\sqrt{3}} \int_0^{1/(1+z)} \frac{da}{a^2H(a) \sqrt{1+3 a \Omega_b/4\Omega_\gamma}}
\end{equation}

For some reason, `cosmomc` uses an approximate formulae instead:
\begin{equation}
\label{eq:9}
  r^\text{approx}_s(z) = \frac{c}{\sqrt{3}} \int_0^{1/(1+z)} \frac{da}{a^2H(a) \sqrt{1+30000 a \Omega_b}}
\end{equation}

The redshift of last scattering is approximated using the fit formula
given in Eq. E-1 in [@hu_sugiyama:1996], and the drag epoch is taken
from Eq.4 in [@Eisenstein_Hu:1997]. We also implement the direct fit
formula for the comoving sound horizon size at drag epoch used in
Eq. 2.5 of [@DESIDRI:VI].

# Numerical results
\label{sec:numerical_results}

## Accuracy

To assess the numerical accuracy of our baseline distance computation,
we compared its results with those from the same $\chi(u)$ integral
evaluated at 10-fold higher resolution, using $10^4$ equally spaced
points over the interval $0 \leq u \leq 0.02$. The difference is
displayed in \autoref{fig:accuracy} for the baseline Planck
$\Lambda$CDM model, reported in Table 1 in [@planck2018VI]. The
difference in distance modulus between the coarse and fine resolution
computation is smaller than $10^{-4}$ mag over the redshift range
$0.01 < z < 1000$, dominated by the interpolation error.

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
number of redshifts requested. Generally, the computation time is
dominated by precomputation steps and remains largely independent of
vector size, except in the case of `astropy` and `jax_cosmo`. We
differentiate between the first call and subsequent calls, as the
initial call may involve specific overheads. For `cosmologix`, this
includes JIT-compilation times, which introduces a significant
delay. Efforts are underway to optimize this aspect. Note that we did
not yet manage to jit-compile the luminosity distance computation in
`cosmoprimo`, due to a compilation error. The speed measurement may
change significantly when this issue is solved.

While `cosmologix` overperforms all other tested codes by a significant
margin in subsequent calls, specific efforts must be taken to avoid
triggering recompilation in order to benefit from this improvement.

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

The JAX implementation enables seamless utilization of hardware accelerators, such as GPUs. However, the CPU-based computation is already highly efficient. To maintain high accuracy in distance calculations, double-precision floating-point arithmetic is currently required, which may necessitate adjustments to fully leverage GPU performance benefits. Given limited motivation to pursue further optimization, we conducted only minimal GPU testing, which indicated that the code, in its present form, does not gain significant performance advantages from GPU execution.

# Differentiability and likelihood maximization

Last, the code provides a framework to efficiently build frequentist
confidence contours for cosmological parameters for all measurements
whose likelihood can be expressed as a chi-square.

To provide the confidence region for a 1 or 2 dimensionnal subset
$\alpha$ of the parameters, we divide the parameter vector as $\theta
= (\alpha, \beta)$ into its fixed ($\alpha$) and varied parts
$\beta$. The explored parameter space is divided on a regular grid,
and for each point $\tilde \alpha$ of the grid, the profile likelihood
$L_p(\beta) = L(\tilde \alpha, \beta)$ needs to be minimized. We take
advantage of the quadratic nature of the chi-square to perform the
minimization using the Gauss-Newton algorithm, which requires only the
function returning the standardized residuals $R(\beta)$ and its
Jacobian $J$ to perform efficient second-order optimization,
approximating the chi-square Hessian as $J^T J$. Two details are
instrumental in speeding up this computation.

Firstly we offset the jit-compilation overhead by forming the restricted
function $R(\beta, \alpha)$ outside the exploration loop, so that the
jit-compilation of $R$ and its jacobian with respect to $\beta$ is
performed only once and used for all the subsequent
minimizations. This is obtained by passing the residual function $R$ to the `gauss_newton_prep` function, defined as follows:
```python
def partial(func, param_subset):
    def _func(x, point):
        return func(dict(x, **point))
    return _func

def gauss_newton_prep(func, params_subset):
    f = jax.jit(partial(func, params_subset))
    return f, jax.jit(jax.jacfwd(f))
```
The jacobian is taken with respect to the first parameter of the partial
function but it can still be evaluated for any specific point of the second set of parameters.

Secondly, we prune the exploration of the grid by starting at the grid
point closest to the global best-fit. We pursue the exploration in
every direction stopping at a $\Delta \chi^2$ threshold. Provided the
threshold is sufficiently high that the resulting region is connected
and encompasses the requested confidence level, the resulting confidence
region is accurate and obtained at a fraction of the cost of an
exhaustive exploration.

\autoref{sample_contour} provides an example 2-dimensionnal
confidence region in the plane $(\Omega_{bc}, w)$ for a flat $w$-CDM
model as probed by the Union3 supernovae compilation
[@2023arXiv231112098R]. The full computation took 3.86s on an
Intel(R) Core(TM) i7-1165G7 at 2.80GHz without GPU acceleration.


![Confidence region at 68 and 95 percent for the $w$ and $\Omega_{bc}$ parameters probed by the Union3 compilation.\label{sample_contour}](sample_contour.pdf)

# References
