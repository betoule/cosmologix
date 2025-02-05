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
\begin{align}
  \label{eq:27}
  ds^2 &= -c^2dt^2 + R^2(t) \left(\frac{dr^2}{1-kr^2} + r^2(d\theta^2 +
    sin^2\theta d\phi^2) \right) \quad \text{with} \quad k = \lbrace-1, 0, 1\rbrace\\
    &= -c^2 dt^2 +  a^2(t)\left( d\chi^2 + R_0^2 S^2\left(\frac{\chi}{R_0}\right) d\Omega^2\right) \quad \text{with } a = \frac{R}{R_0}, S=\left\lbrace
    \begin{array}{l}
      \sinh \text{ if k = 1}\\
      Id \text{ if k = 0}\\
      \sin \text{ if k = -1}\\
    \end{array}
  \right.,
  r = S\left(\frac{\chi}{R_0}\right)
\end{align}
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
