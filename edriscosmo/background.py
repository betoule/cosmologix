import jax.numpy as jnp
from jax import lax
import jax
from typing import Callable, Tuple
from tools import safe_vmap, linear_interpolation, trapezoidal_rule_integration, Constants, restrict


""" Implementation of neutrinos and dark radiation

See Planck 2013 results XVI sect. 2.1.1 and Camb notes
downloadable at http://camb.info/readme.html.
See https://arxiv.org/pdf/astro-ph/0203507

Assumptions in code below are the following:
- Fermi-Dirac distribution for neutrinos
- decoupling while relativistic
- three active neutrinos
if mnu is non zero, a single eigenstate of mass mnu is
considered.

share_delta_neff = T
"""


def _nu_dlnam():
    return -(jnp.log(Constants.am_min / Constants.am_max)) / (Constants.N - 1)


def Omega_nu_rel_per_nu(Omega_gamma):
    return 7. / 8 * (4. / 11) ** (4. / 3) * Omega_gamma

def Omega_nu_h2(params):
    omgh2 = Tcmb_to_Omega_gamma(params["Tcmb"], params["H0"]) * (params["H0"]/100)**2
    neutrino_mass_frac = 1/(omgh2 * (45 * Constants.zeta3)/(2 * jnp.pi**4) * (Constants.e/(Constants.k * params["Tcmb"])) * (4./11.))
    return params["m_nu"] / neutrino_mass_frac * (params["Neff"] / 3)

def Omega_n(params, u):
    """ Reduced density contribution of neutrinos at u

    u = 1/sqrt(1+z)

    To be counted with a scale factor **4 contribution to Hubble rate.
    """
    return Omega_n_rel(params) + Omega_n_mass(params, u)


def Omega_n_rel(params):
    """ Reduced density contribution of relativistic neutrinos
    """
    omg = Tcmb_to_Omega_gamma(params["Tcmb"], params["H0"])
    omr = Omega_nu_rel_per_nu(omg)
    omrnomass = omr * (params["Neff"] - 3.046 / 3.)   ### WRONG IF MASSLESS NEUTRINOS????
    return omrnomass

@safe_vmap(in_axes=(None, 0))
def Omega_n_mass(params, u):
    """ Reduced density contribution of massive neutrinos at u

    u = 1/sqrt(1+z)

    To be counted with a scale factor **4 contribution to Hubble rate.
    Implementation from Nu_rho CAMB routine:
    !  nu_masses=m_nu(i)*c**2/(k_B*T_nu0).
    !  Get number density n of neutrinos from eq.31 astro-ph/0203507
    !  rho_massless/n = int q^3/(1+e^q) / int q^2/(1+e^q)=7/180 pi^4/Zeta(3)
    !  then m = Omega_nu/N_nu rho_crit /n  with N_nu number of massive neutrinos
    the routine computes the
    reduced mass mbar = m_nu c^2 / (kB T_nu) deduced from total Omega_nu_h2
    considering that only one neutrino is massive, so counting (Neff/3)
    with respect to relativistic density
    """
    omrmass = 0
    Omega_nu = Omega_nu_h2(params) / ((params["H0"] / 100) ** 2)
    Omega_massless_per_nu = Omega_nu_rel_per_nu(Tcmb_to_Omega_gamma(params["Tcmb"], params["H0"]))
    Nu_mass_fractions = 1
    Nu_mass_degeneracies = params["Neff"] / 3.
    rho_massless_over_n = Constants.const / (1.5 * Constants.zeta3)
    nu_mass = (Nu_mass_fractions / Nu_mass_degeneracies) * rho_massless_over_n * Omega_nu / Omega_massless_per_nu
    omrmass += Omega_massless_per_nu * nu_rho(u ** 2 * nu_mass)
    return omrmass

#Omega_n_mass = jax.vmap(Omega_n_mass, in_axes=(None, 0))

def nu_rho(am):
    """ Density of one eigenstate of massive neutrinos

    in units of the mean density of one flavor of massless neutrinos.
    am: scale factor times reduced mass
    """
    # comoving momentum in units of k_B * T_nu0/c
    adq = Constants.qmax / float(Constants.nq)
    q = jnp.arange(adq, Constants.qmax + adq, adq)
    aq = am / q
    v = 1 / jnp.sqrt(1 + aq ** 2)
    aqdn = q ** 3 / (1 + jnp.exp(q))

    drhonu = aqdn / v
    rhonu = jnp.sum(drhonu) * adq / Constants.const
    return rhonu


def nu_rho_interp(am): #, nu_params):
    """ Massive neutrino density in unit of mean density of one
    eigenstate of massles neutrino

    # am: reduced mass m_nu / k_B T times scale factor

    fast interpolation in the grid
    """
    am = jnp.asarray(am)
    rhonu = jnp.exp(linear_interpolation(am, Constants.NU_LOG_RHO_TAB, Constants.NU_AM_TAB))

    indexlow = am <= Constants.am_min
    rhonu = rhonu.at[indexlow].set(1 + Constants.const2 * am[indexlow] ** 2)

    indexhigh = am >= Constants.am_max
    # 0.5 or 0.75 ???  0.75 like in pycosmo note and https://arxiv.org/pdf/astro-ph/0203507
    rhonu = rhonu.at[indexhigh].set(
        1.5 / Constants.const * (Constants.zeta3 * am[indexhigh] + 0.75 * 15 * Constants.zeta5 / am[indexhigh]))
    return rhonu


def binned_cosmo(params, explanatory):
    '''Approximation of the distance modulus-redshift relation as a
    1st order B-spline (Piece-wise linear function).
    
    Parameters:
    -----------
    params: a jax pytree providing values for 
           - mu_bins (n_bins,)
    explanatory: pytree explanatory (independent) variables for the events. Typically:
            - z (N,) the redshift
            - z_bins (n_bins), the redshift nodes

    '''
    return linear_interpolation(jnp.log10(explanatory['z']), params['mu_bins'], jnp.log10(explanatory['z_bins']))


##############
# Background
##############
def rhoc(H):
    """ Return the critical density (in kg/m3)

    Parameters
    ----------
    H: Hubble parameter in km/s/Mpc
    """
    return 3 * (H * 1e-3 / Constants.pc) ** 2 / (8 * jnp.pi * Constants.G)


def Omega_c(params):
    h = params["H0"] / 100
    Omega_b0 = params['Omega_b_h2'] / h ** 2
    Omega_nu_mass = Omega_n_mass(params, jnp.atleast_1d(1.))[0]
    return params['Omega_m'] - Omega_b0 - Omega_nu_mass


def Omega_de(params, Omega_n_rel):
    omg = Tcmb_to_Omega_gamma(params["Tcmb"], params["H0"])
    return 1 - params['Omega_m'] - omg - Omega_n_rel - params['Omega_k']


def dzoveru3H(params, u):
    """ Better integrand for inverse expansion rate integrals
    
    Return (1+z)^{-3/2} H0/H(z)
    """
    h = params["H0"] / 100
    Omega_b0 = params['Omega_b_h2'] / h ** 2
    Omega_c0 = Omega_c(params)
    #Omega_nu_mass = jnp.array([Omega_n_mass(params, uu) for uu in u])
    Omega_gamma = Tcmb_to_Omega_gamma(params["Tcmb"], params["H0"])
    Omega_nu_mass = Omega_n_mass(params, u)
    Omega_nu_rel = Omega_n_rel(params)
    Omega_de0 = Omega_de(params, Omega_nu_rel)
    return 1. / jnp.sqrt(  # as in CCL or PyCosmo
        (Omega_c0 + Omega_b0) +
        Omega_de0 * u ** (-6 * params['w']) +
        (Omega_gamma + Omega_nu_rel + Omega_nu_mass) * u ** (-2) +
        params['Omega_k'] * u ** 2
    )


def dC(params, z, nstep=300.):
    """Return the comoving distance at redshift z
    
    Distance between comoving object and observer that stay
    constant with time (coordinate).

    For some reason (c was expressed in m/s and H0 in km/s/Mpc in
    Constants.py) this function return distances in kpc. I you
    change that, you will have the satisfaction of doing something
    right, but so many things will break down in my codes that I
    wont be able to survive.

    Parameters:
    -----------
    params: pytree containing the background cosmological parameters
    z: scalar or array
       redshift at which to compute the comoving distance

    Returns:
    --------
    Comoving distance in Mpc
    """
    dh = Constants.c / params['H0'] * 1e-3  # Hubble radius in kpc
    u = 1 / jnp.sqrt(1 + z)
    umin = 0.02  #u.min()
    step = (1 - umin) / nstep
    _u = jnp.arange(umin + 0.5 * step, 1, step)
    csum = jnp.cumsum(dzoveru3H(params, _u[-1::-1]))[-1::-1]
    R = linear_interpolation(u, csum, _u - 0.5 * step) \
        * 2 * step * dh
    return R


def dM(params, z):
    """Return the transverse comoving distance in Mpc.
    
    The comoving distance between two comoving objects (distant
    galaxies for examples) separated by an angle theta is dM
    theta.
    """
    comoving_distance = dC(params, z)
    index = - jnp.sign(params["Omega_k"]).astype(jnp.int8) + 1
    dh = Constants.c / params['H0'] * 1e-3  # Hubble distance in kpc
    sqrt_omegak = jnp.sqrt(jnp.abs(params['Omega_k']))

    def open(com_dist):
        return (dh / sqrt_omegak) * jnp.sinh(sqrt_omegak * com_dist / dh)

    def flat(com_dist):
        return com_dist

    def close(com_dist):
        return (dh / sqrt_omegak) * jnp.sin(sqrt_omegak * com_dist / dh)

    branches = (open, flat, close)
    return lax.switch(index, branches, comoving_distance)


def dL(params, z):
    """Luminosity distance in Mpc.
    """
    return (1 + z) * dM(params, z)
    #return (1 + z) * dC(params, z)


def dA(params, z):
    """Angular diameter distance in Mpc.
    
    The physical proper size of a galaxy which subtend an angle
    theta on the sky is dA * theta
    
    """
    return dM(params, z) / (1 + z)


def mu(params, z):
    """ Distance modulus
    """
    return 5 * jnp.log10(dL(params, z)) + 25


def dVc(params, z):
    """ Differential comoving volume in Mpc^3

    Notes:
    ------

    Pretty much equivalent to:
        1. create a Vc function to compute comoving volume such as:
         
            Vc = (4/3) * jnp.pi * dC(params, z)**3

           and computing/evaluating dVc(z) with jax.grad

        2. astropy.cosmology.FlatLambdaCDM.differential_comoving_volume(z).value * 4 * jnp.pi

    """
    dh = Constants.c / params['H0'] * 1e-3
    toto = 1 + z
    return 4 * jnp.pi * (dC(params, z) ** 2) * dh * dzoveru3H(params, 1 / jnp.sqrt(toto)) / (toto ** (3 / 2))


def Tcmb_to_Omega_gamma(Tcmb, H0):
    rhogamma = 4 * Constants.sigma * Tcmb ** 4 / Constants.c ** 3  # Energy density of CMB
    return rhogamma / rhoc(H0)


###########################
# Accoustic scale
###########################
#
# Approximation for z_star and z_drag
#
def z_star(params):
    """ Redshift of the recombination
    """
    Obh2 = params['Omega_b_h2']
    h2 = params['H0'] ** 2 * 1e-4
    odm = params['Omega_m']
    g1 = 0.0783 * Obh2 ** -0.238 / (1 + 39.5 * Obh2 ** 0.763)
    g2 = 0.560 / (1 + 21.1 * Obh2 ** 1.81)
    return 1048 * (1 + 0.00124 * Obh2 ** -0.738) * \
        (1 + g1 * (odm * h2) ** g2)


def z_drag(params):
    """ Redshift of the drag epoch
    
    Fitting formulae for adiabatic cold dark matter cosmology.
    Eisenstein & Hu (1997) Eq.4, ApJ 496:605
    """
    omegamh2 = params['Omega_m'] * (params['H0'] * 1e-2) ** 2
    b1 = 0.313 * (omegamh2 ** -0.419) * (1 + 0.607 * omegamh2 ** 0.674)
    b2 = 0.238 * omegamh2 ** 0.223
    return 1291 * (1. + b1 * params['Omega_b_h2'] ** b2) * \
        omegamh2 ** 0.251 / (1 + 0.659 * omegamh2 ** 0.828)


def a4H2(params, a):
    """Return a**4 * H(a)**2/H0**2
    """
    h = params["H0"] / 100
    Omega_b0 = params['Omega_b_h2'] / h ** 2
    Omega_c0 = Omega_c(params)
    Omega_nu_mass = jnp.array([Omega_n_mass(params, jnp.sqrt(aa)) for aa in a])
    Omega_nu_rel = Omega_n_rel(params)
    Omega_de0 = Omega_de(params, Omega_nu_rel)
    Omega_gamma = Tcmb_to_Omega_gamma(params["Tcmb"], params["H0"])
    return ((Omega_b0 + Omega_c0) * a + params["Omega_k"] * (a ** 2) +
            Omega_gamma + Omega_nu_rel + Omega_nu_mass +
            Omega_de0 * a ** (1 - 3. * params['w']))


def dsound_da_approx(params, a):
    """Approximate form of the sound horizon used by cosmomc

    Notes
    -----

    This is to be used in comparison with values in the cosmomc chains

    """
    return 1. / (jnp.sqrt(
        a4H2(params, a) * (1. + 3e4 * params['Omega_b_h2'] * a))
                 * params['H0'])


def rs(params, z, dsound_da=dsound_da_approx):
    """ The comoving sound horizon size in Mpc
    """
    nstep = 1000
    a = 1. / (1. + z)
    _a = jnp.linspace(0, a, nstep)
    _a = 0.5 * (_a[1:] + _a[:-1])
    step = _a[1] - _a[0]
    #step = a / nstep
    #_a = jnp.arange(0.5 * step, a, step)
    R = Constants.c * 1e-3 / jnp.sqrt(3) * dsound_da(params, _a).sum() * step
    return R


class GeometricCMBPrior():
    def __init__(self, mean, covariance):
        '''An easy-to-work-with summary of CMB measurements

        Parameters:
        -----------
        mean: best-fit values for Omega_ch2, Omega_bh2, and 100tetha_MC

        covariance: covariance matrix of vector mean
        '''
        self.mean = jnp.array(mean)
        self.cov = jnp.array(covariance)
        self.W = jnp.linalg.inv(self.cov)
        self.L = jnp.linalg.cholesky(self.W)

    def model(self, params):
        Omega_c_h2 = Omega_c(params) * (params['H0'] ** 2 * 1e-4)
        zstar = z_star(params)
        rsstar = rs(params, zstar)
        thetastar = rsstar / dM(params, zstar) * 100.
        return jnp.array([Omega_c_h2, params['Omega_b_h2'], thetastar])

    def residuals(self, params):
        return self.mean - self.model(params)

    def weighted_residuals(self, params):
        return self.L @ self.residuals(params)

    def likelihood(self, params):
        r = self.weighted_residuals(params)
        return r.T @ r


planck2018 = GeometricCMBPrior(
    [2.2337930e-02, 1.2041740e-01, 1.0409010e+00],
    [[2.2139987e-08, -1.1786703e-07, 1.6777190e-08],
     [-1.1786703e-07, 1.8664921e-06, -1.4772837e-07],
     [1.6777190e-08, -1.4772837e-07, 9.5788538e-08]])

simu_params = {'Omega_m': 0.3,
               'Tcmb': 2.7255,  # Kelvin, Fixen (2009)
               'Omega_b_h2': 0.02204854,
               'w': -1.,
               'H0': 70.,
               }

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from astropy import cosmology
    import numpy as np
    plt.ion()
    #
    # Distance modulus comparison
    #
    Tcmb = 2.7255  # Kelvin, Fixen (2009)
    cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.30, Tcmb0=Tcmb, Neff=0)
    z = jnp.linspace(0.01, 1)
    params = {'Omega_m': 0.3,
              'Tcmb': Tcmb,
              'Omega_b_h2': 0.02204854,
              'Omega_k': 0.,
              'w': -1.,
              'H0': 70.,
              'm_nu': 0.06,
              'Neff': 3.046
              }

    fig = plt.figure('Comparison')
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(z, mu(params, z), label='edris')
    ax1.plot(z, cosmo.distmod(np.asarray(z)).value, label='astropy', ls='--')
    ax1.legend()
    ax2.plot(z, mu(params, z) - cosmo.distmod(np.asarray(z)).value, label='astropy')
    ax2.set_xlabel(r'$z$')
    ax1.set_ylabel(r'$\mu$ [mag]')
    ax2.set_ylabel(r'$\Delta \mu$ [mag]')



    #
    # Gradients
    #
    params['Omega_k'] = 0.001 # Need to work to get dmu/dOmega_k around 0
    plt.figure('Gradients')
    J = jax.jacobian(mu)
    zbig = jnp.linspace(0.01,1000, 3000)
    G = J(params, zbig)
    for p in params:
        plt.plot(zbig, G[p], label=p)
    plt.yscale('symlog', linthresh=1e-5)
    plt.xscale('log')
    plt.xlabel('z')
    plt.ylabel(r'$\frac{\partial \mu}{\partial \theta}$')
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    
    #
    # Acoustic scale
    #
    zstar = z_star(params)
    rsstar = rs(params, zstar)
    thetastar = rsstar / dM(params, zstar) * 100.
    print(f'{zstar=}, {rsstar=}, {thetastar=}')
    print(f'{planck2018.likelihood(params)}')

    pl = restrict(
        planck2018.likelihood,
        {'Omega_gamma': Tcmb_to_Omega_gamma(Tcmb, 70),
         'w': -1}
    )
    x0 = {'Omega_m': 0.3,
          'Omega_b_h2': 0.02204854,
          'H0': 70.,
          }
    #bf18 = tncg(pl, x0, verbose=True)
    import jax

    f = jax.jit(planck2018.weighted_residuals)
    J = jax.jit(jax.jacobian(planck2018.weighted_residuals))
    plt.show()
