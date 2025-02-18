from cosmologix import densities
from cosmologix.tools import Constants, trapezoidal_rule_integration, safe_vmap
from cosmologix.interpolation import chebyshev_nodes, newton_interp, newton_divided_differences
import jax.numpy as jnp
import jax

def rho_gamma_0(params):
    """ Energy density of CMB photons today in kg/m^3
    """
    return 4 * Constants.sigma * params['Tcmb'] ** 4 / Constants.c**3

def T_nu(params):
    """ Neutrinos distribution temperature today

    See 2005NuPhB.729..221M
    """
    params['T_nu'] = (4/11)**(1./3) * (params['Neff'] / 3)**(1./4) * params['Tcmb']
    return params

def rho_nu_i_relativistic(params):
    return 7./8 * params['Neff'] / 3 * (4/11)**(4./3) * rho_gamma_0(params)

@safe_vmap()
def rho_nu(params, z):
    return rho_nu_i_relativistic(params) * 120 / (7 * jnp.pi**4) * composite_I(params['m_nu_bar'] / (1 + z))

@safe_vmap(in_axes=(0,))
def I_m(m_bar):
    r'''Helping function to compute the integral of the energy
    distribution of massive fermions.

    This function of the reduced mass \bar m evaluates:
    \int_0^\inf x^3 \sqrt(1 + (m_bar/x)^2)/(e^x + 1) dx
    '''
    def integrand(x):
        return x**3 * jnp.sqrt(1+(m_bar/x)**2) / (1 + jnp.exp(x))
    return trapezoidal_rule_integration(integrand, 1e-3, 31,10000)

def m_bar(params):
    ''' Convert neutrinos masses from eV to reduced energy parameter m_bar

    m_bar = m c²/k_b T
    '''
    return jnp.array([params['m_nu'], 0., 0.]) * Constants.e / (Constants.k * params['T_nu'])

def analytical_m_small(m_bar):
    return 7*jnp.pi**4/120*(1+5/(7*jnp.pi**2) * m_bar**2)

def analytical_m_large(m_bar):
    return 3./2 * Constants.zeta3 * m_bar + 3/(4*m_bar) * 15 * Constants.zeta5

# Tabulated functions
_mbar =  jnp.logspace(-2, 2, 1000)
_Imbar = I_m(_mbar)

n_cheb = 35
bar_nodes = chebyshev_nodes(n_cheb, -2, 3)
coeffs = newton_divided_differences(bar_nodes, I_m(10**bar_nodes))
interp2 = lambda x: newton_interp(bar_nodes, None, jnp.log10(x), coeffs=coeffs)

def interpolated_I(m_bar):
    return linear_interpolation(jnp.log(m_bar), _Imbar, jnp.log(_mbar))

# Define the composite function using lax.switch
@safe_vmap(in_axes=(0,))
def composite_I(x):
    # Compute the index based on x
    index = jnp.digitize(x, jnp.array([0.01, 1000]))
    
    # Define branches
    #branches = [analytical_m_small, interpolated_I, analytical_m_large]
    branches = [analytical_m_small, interp2, analytical_m_large]
    
    # Use lax.switch to select the appropriate branch
    return jax.lax.switch(index, branches, x)


if __name__ == '__main__':
    from cosmologix import Planck18
    import matplotlib.pyplot as plt
    from cosmologix.polynomial_interpolation import chebyshev_nodes, barycentric_weights_chebyshev, barycentric_interp

    params = densities.params_to_density_params(Planck18.copy())
    params = T_nu(params)
    params['m_nu_bar'] = m_bar(params)
    plt.ion()
    
    #weights = barycentric_weights_chebyshev(n_cheb)
    #interp = jax.vmap(lambda x: barycentric_interp(bar_nodes, I_m(10**bar_nodes), jnp.log10(x), weights), in_axes=(0,))
    #
    #  
    
    #mbar = jnp.logspace(-3, 4, 1000)
    #mlim = jnp.logspace(-2, 3, 1000)
    #fig = plt.figure('Integral_accuracy')
    #ax1, ax2 = fig.subplots(2,1,sharex=True)
    #ax1.loglog(mbar, I_m(mbar))
    #ax1.loglog(mlim, interp(mlim))
    #ax1.loglog(mlim, interp2(mlim))
    #ax2.plot(mbar, composite_I(mbar)/I_m(mbar))
    #ax2.plot(mlim, interp(mlim)/I_m(mlim))
    #ax2.plot(mlim, interp2(mlim)/I_m(mlim))
    #plt.show()

    z = jnp.linspace(0.01, 1000, 1000)    
    plt.plot(z, rho_nu(params, z))
    plt.show()
