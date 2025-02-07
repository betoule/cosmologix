from cosmologix.likelihoods import DES5yr
from cosmologix import Planck18
from cosmologix.tools import restrict
from cosmologix.fitter import newton

des = DES5yr()
fixed_params = Planck18.copy()
fixed_params.pop('Omega_m')
l = restrict(des.negative_log_likelihood, fixed_params=fixed_params)

bestfit, extra = newton(l, {'Omega_m': 0.3, 'M':0})

