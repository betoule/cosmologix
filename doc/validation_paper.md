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
	orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Kuhn, Dylan
	orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: LPNHE, CNRS, France
   index: 1
   ror: 01hg8p552
 - name: IJCLab, Orsay, France
   index: 2
   ror: 03gc1p724
date: 05 February 2025
bibliography: paper.bib

---

# Summary

Type-Ia supernovae are standardizable candles allowing to measure
distances in the universe. Inference of cosmological parameters from
such datasets is made easy by providing fully differentiable
computation of the distance-redshift relation as function of the
cosmological parameters. We also provide common fitting formulae for
the acoustic scale so that the resulting code can be used for fast
cosmological inference from supernovae in combination with BAO and CMB
distance measurements. We check the accuracy of our computation
against CCL and astropy.cosmology and show that our implementation can
outperform both codes by X order of magnitudes in speed while
maintaining a reasonable accuracy of X mag on distance modulii in the
redshift range $0.01--1000$.

# Statement of need

Several software are available to compute cosmological distances
including astropy, camb, class, ccl. To our knowledge only jax-cosmo
provide automatic differentiation through the use of
JAX. Unfortunately, at the time of writing, the distance computation
in jax-cosmo is neglecting contributions to the energy density from
neutrinos species. The accuracy of the resulting computation is
insufficient for the need of the LEMAITRE analysis.

The LEMAITRE collaboration is therefore releasing its internal code
for the computation of cosmological distances. The computation itself
is very standard, but the implementation in JAX is taylored for speed,
while preserving reasonable accuracy.

# Computations of the homogeneous background evolution

## Friedmann equations

All computations in cosmologix are made for the
Friedman-Lemaitre-Robertson-Walker metric (isotropic and homogeneous
universe).
\begin{equation}
  \label{eq:27}
  ds^2 = -c^2dt^2 + R^2(t) \left(\frac{dr^2}{1-kr^2} + r^2(d\theta^2 +
    sin^2\theta d\phi^2) \right) \quad \text{with} \quad k = \lbrace-1, 0, 1\rbrace.
\end{equation}
Denoting $a = \frac{R}{R_0}$ and:
$$
S=\left\lbrace
    \begin{array}{l}
      \sinh \text{ if k = 1}\\
      Id \text{ if k = 0}\\
      \sin \text{ if k = -1}\\
    \end{array}
  \right.,
$$
and $r = S\left(\frac{\chi}{R_0}\right)$, the metrics rewrites:
\begin{equation}
    ds^2 = -c^2 dt^2 +  a^2(t)\left( d\chi^2 + R_0^2 S^2\left(\frac{\chi}{R_0}\right) d\Omega^2\right)
\end{equation}

The first Friedman equation without the cosmological
constant term (whose role is held by the fluid) reads:

\begin{equation}
  \label{eq:2}
  H^2 = \frac{8\pi G}{3}\rho - \frac{k}{R^2}\,,
\end{equation}

where $\rho$ is the proper energy density, $R$ the scale factor, $H =
\dot R / R$ the Hubble parameter and $k = {-1, 0, 1}$ is the sign of
spatial curvature. The value of constants (such as $G$) used in the
code are given in Table

Denoting, as usual, $\rho_{crit} = \frac{3 H_0^2}{8\pi G}$ the current
value of the critical density, $\Omega_0 = \frac{\rho_0}{\rho_{crit}}$ the
reduced energy density today and $\Omega_k = -\frac{k}{R^2_0H^2_0}$, one
can rewrite equation under its most common form:
	\begin{equation}
  \label{eq:3}
  \frac{H^2}{H_0^2} = \Omega_0 \frac{\rho}{\rho_0} + \Omega_k (1+z)^2
\end{equation}

We split the universe content into several components characterized by
there reduced density today and their equation of state.  For a
perfect fluid $X$ with equation of state $\rho_X = w_x p_x$, the
energy conservation writes:

\begin{equation}
  \label{eq:4}
  \frac{d}{dt}(\rho_x R^3) = -3 p_x R^2 \dot R = -3 w_x \rho_x R^2 \dot R\,,
\end{equation}
which can be integrated to give:

\begin{equation}
  \label{eq:5}
  \log \frac{\rho_x}{\rho_x^0} = 3 \int_0^z \frac{1+w_x(z)}{1+z}dz
\end{equation}

In the code we consider the following components:
- Baryonic matter: $\Omega_b, w_b=0 \rightarrow \frac{\rho_b}{\rho_b^0} = (1+z)^3$,
- Cold dark matter: $\Omega_c, w_c=0 \rightarrow \frac{\rho_b}{\rho_b^0} = (1+z)^3$,
- Photons: $\Omega_\gamma(T_\text{cmb}, h), w_\gamma=\frac13 \rightarrow
  \frac{\rho_\gamma}{\rho_\gamma^0} = (1+z)^4$, 
- Neutrinos, $\Omega_n$,
- and dark energy $\Omega_x$.

## Dark energy

Varying EoS: $w(z) = w + \frac{z}{1+z} w_a \rightarrow \rho/\rho_0 =
  \exp\left(3 (1 + w + w_a) \log(1+z) - 3 w_a \frac{z}{1+z}\right)$,
  with $w$ and $w_a$ as a free parameter.
  
## Radiation

The energy density of specie $i$ is obtained by integration over the
distribution function:
\begin{equation}
  \label{eq:12}
  \rho_i c^2= g_i  \int N_i(p) E(p) \frac{4\pi p^2dp}{h^3}
\end{equation}
where $g_i$ is the degeneracy number of the species. 

### Photon

For photons with two spin states:
\begin{equation}
  \label{eq:14}
  N_\gamma(p) = \frac{1}{e^{\frac{cp}{k_B T_\gamma}} + 1}, \quad g_i=2,
\end{equation}
which gives, with the variable change $x=\frac{cp}{k_BT_\gamma}$:
\begin{align}
  \label{eq:15}
  \rho_\gamma c^2 &= \frac{8\pi(k_B T_\gamma)^4}{ c^3 h^3}  \int_0^\infty \frac{x^3}{e^x - 1} dx\\
  &=\frac{8\pi^5(k_B T_\gamma)^4}{15 c^3 h^3} 
\end{align}
where we have used the result $\int_0^\infty u^{s-1}/(e^u - 1) du =
\Gamma(s) \zeta(s)$. The photon density today is fixed in the code
from $T_\gamma^0 = 2.7255 K$ 2009ApJ...707..916F.

### Neutrinos

For neutrinos following the Fermi-Dirac distribution, neglecting
the chemical potential at high temperature, we have:
\begin{equation}
  \label{eq:11}
  N_i(p) = \frac{1}{e^{\frac{E}{k_B T_i}} + 1}
\end{equation}
with $E(p)^2 = c^2p^2 + m^2c^4$. While relativistic, with the same
variable change as above:
\begin{align}
  \label{eq:13}
  \rho^\text{nomass}_i c^2&= g_i \frac{4\pi(k_B T_i)^4}{ c^3 h^3}  \int_0^\infty \frac{x^3}{e^x + 1} dx\\
  &=\frac{g_i 4\pi^5(k_B T_i)^4}{15 c^3 h^3}\frac{7}{8}
\end{align}
where we have used $\int_0^\infty u^{s-1}/(e^u + 1) du = \Gamma(s)
\zeta(s) (1-1/2^{s-1})$. The universe is heated by the
electron-positron annihilation when the decoupling of neutrinos is
nearly complete. The neutrinos and photon temperature after
annihilation are related by:
\begin{equation}
  \label{eq:16}
  \frac{T_\nu}{T_\gamma} = \left( \frac{4}{11}\right)^{1/3} \left(\frac{N_\text{eff}}3\right)^{1/4}
\end{equation}
with $N_\text{eff} = 3.046$ {2005NuPhB.729..221M}, which gives an
effective density for 6 species of relativistic neutrinos and
anti-neutrinos:
\begin{equation}
  \label{eq:18}
  \rho_\nu = \frac78 N_\text{eff}\left(\frac{4}{11}\right)^{4/3}\rho_\gamma\,.
\end{equation}

### Massive neutrinos

The distribution of neutrinos after decoupling is frozen. The energy
density for massive neutrinos, assuming that the decoupling occurs
when neutrinos are still ultra-relativistic, is given by:
\begin{align}
  \label{eq:17}
  \rho_i & = g_i \frac{4\pi(k_B T_i)^4}{ c^3 h^3}  \int_0^\infty \frac{x^3\sqrt{1 + (\bar m/x)^2}}{e^x + 1} dx\\
  & = \rho^\text{nomass}_i \frac{120}{7\pi^4} I(\bar m)\,,
\end{align}
where we denote $\bar m = m_i c^2 / (k_B T_i)$.  In practice our
implementation mimic the CAMB code\footnote{routine Nu\_rho}:
\begin{itemize}
\item ultra-relativistic $\bar m \leq 0.01$: $I(\bar m) \sim I(0) (1 + \frac5{7\pi^2} \bar m^2)$.
\item intermediate $0.01 < \bar m < 600$: $I(\bar m)$ is evaluated
  numerically using the Simpson rule in a log grid of $\bar m$. Values
  are interpolated linearly in the grid.
\item non-relativistic $\bar m \geq 600$: $I(\bar m) \sim \frac32 \zeta(3) \bar m + \frac{3}{4\bar m} 15 \zeta(5)$ 
\end{itemize}

The baseline assumption in the Planck 2013 release is to postulate 3
neutrinos with $N_\text{eff}=3.046$ and one massive eigenstate with
$m_\nu = 0.06 {\rm eV}$, which we follow exactly to enable the direct
use of their results. So to summarize, I use \emph{(The counting is still unclear to me but I think this is what matches the use in Planck)}:
\begin{itemize}
\item massless case: $\sum g_i = 6$,
\item massive case: $\sum_{massless} g_i = 4$, $\sum_{massive} g_i = 2$ with $m_\nu = 0.06 {\rm eV}$.
\end{itemize}

Note that we count:
\begin{align}
  \label{eq:19}
  \Omega_\nu h^2 &= \frac{n_\nu * m_\nu}{\rho_{crit}^0} = \frac{\rho^0_\gamma}{\rho_{crit}^0}h^2 \frac{45}{2\pi^4} \zeta(3) \frac{e}{k_B T_\gamma}\frac{4}{11} \left(\frac{N_\text{eff}}{3}\right)^{(3/4)} \sum \frac{g_i}{2}\frac{m_i}{\rm eV}\\
  &= \left(\frac{N_\text{eff}}{3}\right)^{(3/4)} \sum \frac{g_i}{2}\frac{m_i}{94.073{\rm eV}}\\
  &= 6.45 \cdot 10^{-4}\\
\end{align}
in $\Omega_m = \Omega_b + \Omega_c + \Omega_\nu$.

## Distances

\label{sec:distances}
In the code, we denote:
\begin{equation}
  \label{eq:28}
  D_h = \frac{c}{H_0}
\end{equation}
The comoving distance (coordinate) is:
\begin{equation}
  \label{eq:25}
  \chi(z) = D_h \int_o^z \frac{dz}{H/H_0}
\end{equation}
The transverse comoving distance is:
\begin{equation}
  \label{eq:1}
  D(z) = D_h \vert\Omega_k\vert^{-1/2} S\left(\vert\Omega_k\vert^{1/2} \frac{\chi}{D_h}\right)\,.
\end{equation}
The integral is much easier to evaluate numerically with the following change of variable $u = (1+z)^{-1/2}$:
\begin{equation}
  \label{eq:6}
  D(u) = 2 D_h \vert\Omega_k\vert^{-1/2} S\left(\vert\Omega_k\vert^{1/2} \int_u^1 \frac{du}{u^3H/H_0}\right)\,.
\end{equation}
The integrand in the last equation is:
\begin{equation}
  \label{eq:7}
  (u^3H/H_0)^{-1} = \left(\Omega_b + \Omega_c + \Omega_k u^2 + (\Omega_\gamma+\Omega_n) u^4 + \Omega_x e^{-6(w+w_a)\log u + 3 w_a (u^2-1) } \right)^{-1/2}
\end{equation}

## Volume
\label{sec:volume}

The elementary comoving volume defined by an elementary redshift slice
$dz$ at redshift $z$ and solid angle $d\Omega$ is given by:
\begin{equation}
  \label{eq:24}
  dV = d\chi D(z)^2 d\Omega = D_h \frac{dz}{H(z)/H_0} D(z)^2 d\Omega
\end{equation}
Integration between redshifts $0$ and $z$ for a flat universe is easy and gives:
\begin{equation}
  \label{eq:26}
  V(z) = d\Omega \int_0^{\chi(z)}\chi^2 d\chi = \frac{\chi^3d\Omega }{3}
\end{equation}
For a non flat universe, a bit of trigonometric manipulation gives:
\begin{equation}
  \label{eq:26}
  V(z) = \frac{d\Omega D_h^2}{\vert \Omega_K\vert}\int_0^{\chi(z)} S^2\left(\sqrt{\vert \Omega_k\vert}\frac{\chi}{D_h}\right) d\chi = \frac{d\Omega D_h^2}{2 \Omega_k} \left[D(z) \sqrt{1 + \Omega_k \left(\frac{D(z)}{D_h}\right)^2} - \chi\right]
\end{equation}

## Sound horizon
\label{sec:sound-horizon}

We also need the sound horizon at a given redshift:
\begin{equation}
  \label{eq:8}
  r_s(z) = \frac{c}{\sqrt{3}} \int_0^{1/(1+z)} \frac{da}{a^2H(a) \sqrt{1+3 a \Omega_b/4\Omega_\gamma}}
\end{equation}
For some reason, cosmomc uses an approximate formulae instead:
\begin{equation}
\label{eq:9}
  r_s(z) = \frac{c}{\sqrt{3}} \int_0^{1/(1+z)} \frac{da}{a^2H(a) \sqrt{1+30000 a \Omega_b}}
\end{equation}
and one has:
\begin{equation}
  \label{eq:10}
  a^4 * \frac{H^2}{H_0^2} = (\Omega_c+\Omega_b) a + (\Omega_\gamma+\Omega_n) + \Omega_k a^2 + \Omega_x a^{1 - 3 (w+w_a)}e^{3w_a(a-1)}
\end{equation}

# Numerical results


# Differentiability and likelihood maximization


# Citations



# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We thank Mickael Rigault for pushing for the public release of this code and coming up with the name.

# References
