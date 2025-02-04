'''
Fitting formulae for the acoustic scale
'''

from typing import Callable, Tuple, Dict

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
