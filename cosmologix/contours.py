from cosmologix.likelihoods import LikelihoodSum
from cosmologix.fitter import restrict_to, restrict, newton, partial, flatten_vector, newton_partial, gauss_newton_partial, gauss_newton_prep, unflatten_vector
from cosmologix.tools import conflevel_to_delta_chi2
from cosmologix import Planck18
import jax.numpy as jnp

def chi2_grid(likelihoods,
              grid={'Omega_m': [0.18,0.48,30],
                    'w': [-0.6, -1.5,30]},
              varied=[],
              fixed=None,
              ):
    likelihood = LikelihoodSum(likelihoods)

    # Update the initial guess with the nuisance parameters associated
    # with all involved likelihoods
    params = likelihood.initial_guess(Planck18)
    if fixed is not None:
        params.update(fixed)
        #loss = restrict(likelihood.negative_log_likelihood, fixed)
        wres = restrict(likelihood.weighted_residuals, fixed)
        initial_guess = params.copy()
        for p in fixed:
            initial_guess.pop(p)
    else:
        #loss, initial_guess = restrict_to(likelihood.negative_log_likelihood, params, varied=list(grid.keys())+varied, flat=False)
        wres, initial_guess = restrict_to(likelihood.weighted_residuals, params, varied=list(grid.keys())+varied, flat=False)
    # Looking for the global minimum
    #bestfit, extra = newton(loss, initial_guess, tol=1e-3, niter=10)
    #minchi2 = extra['loss'][-1]
    wres_, J = gauss_newton_prep(wres, initial_guess)
    x0 = flatten_vector(initial_guess)
    xbest, extra = gauss_newton_partial(wres_, J, x0, {})
    bestfit = unflatten_vector(initial_guess, xbest)
    # Exploring the chi2 space
    explored_params = list(grid.keys())
    grid_size = [grid[p][-1] for p in explored_params]
    chi2_grid = jnp.full(grid_size, jnp.nan)
    x_grid, y_grid = [jnp.linspace(*grid[p]) for p in explored_params]

    partial_bestfit = bestfit.copy()
    for p in explored_params: partial_bestfit.pop(p)
    
    x = flatten_vector(partial_bestfit)
    #partial_loss, g, H = partial(loss, partial_bestfit)
    wres_, J = gauss_newton_prep(wres, partial_bestfit)
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            print(i, j)
            point = {explored_params[0]:x_grid[i], explored_params[1]:y_grid[j]}
            #x, ploss = newton_partial(partial_loss, x, g, H, point, niter=10)
            x, ploss = gauss_newton_partial(wres_, J, x, point)
            chi2_grid = chi2_grid.at[i, j].set(ploss['loss'][-1])
    return {'params': explored_params,
            'x': x_grid,
            'y': y_grid,
            'chi2': chi2_grid,
            'bestfit': bestfit,
            'extra': extra}


#    # Same but using gauss newton
#    def wres(params):
#        return jnp.hstack([l.weighted_residuals(params) for l in likelihoods])
#    _wres = restrict(wres, fixed_params=fixed_params)
#    res = lambda x, fixed: _wres(unflatten_vector(initial_guess, x))
#    jac = jax.jacfwd(res)
#
#    bestfit2, extra2 = gauss_newton_partial(res, jac, flatten_vector(initial_guess), {})
#    bestfit2 = unflatten_vector(initial_guess, bestfit2)
#

#    # Same thing with gauss_newton
#    @jax.jit
#    def part_wres(x, grid_point):
#        return _wres(dict(unflatten_vector(partial_bestfit, x), **grid_point))
#    
#    part_jac = jax.jacfwd(part_wres)
#    chi2_grid2 = jnp.full(grid_size, jnp.nan)
#    x = flatten_vector(partial_bestfit)
#    for i in range(grid_size[0]):
#        for j in range(grid_size[1]):
#            print(i, j)
#            point = {ex_params[0]:x_grid[i], ex_params[1]:y_grid[j]}
#            x, extra = gauss_newton_partial(part_wres, part_jac, x, point)
#            chi2_grid2 = chi2_grid2.at[i, j].set(extra['loss'][-1])

#    import matplotlib.pyplot as plt
#    plt.ion()
#    plt.contour(x_grid, y_grid, chi2_grid.T - chi2_grid.min(), conflevel_to_delta_chi2(jnp.array([68.3,95.5])))
#    plt.plot(bestfit['Omega_m'], bestfit['w'], 'r+')
#
# 
#    plt.show()
#

def plot_contours(grid, ax=None):
    x, y = grid['params']
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    ax.contour(grid['x'], grid['y'], grid['chi2'].T - grid['chi2'].min(), [conflevel_to_delta_chi2(l) for l in jnp.array([68.3,95.5])])
    ax.plot(grid['bestfit'][x], grid['bestfit'][y], 'rs')

if __name__ == '__main__':
    import jax
    from cosmologix.likelihoods import DES5yr, Planck2018Prior
    jax.config.update("jax_enable_x64", True)
    #bestfit, extra = chi2_grid([DES5yr(), Planck2018Prior()], varied=['M', 'Omega_b_h2', 'H0'])
    fixed = Planck18.copy()
    for p in ['Omega_b_h2', 'H0', 'w', 'Omega_m']: fixed.pop(p)
    grid = chi2_grid([DES5yr(), Planck2018Prior()], fixed=fixed)
    #varied=['M', 'Omega_b_h2', 'H0'])
    #grid = chi2_grid([DES5yr()], varied=['M'])
    plot_contours(grid)
    import matplotlib.pyplot as plt
    plt.show()
    
