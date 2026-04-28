"""Best fit cosmologies."""

AVAILABLE_PARAMETER_SETS = {
    # Base-ΛCDM cosmological parameters from Planck
    # TT,TE,EE+lowE+lensing. Taken from Table 1. in
    # 10.1051/0004-6361/201833910
    "Planck18": {
        "Tcmb": 2.7255,  # from Planck18 arxiv:1807.06209 footnote 14 citing Fixsen 2009
        "Omega_bc": (0.02233 + 0.1198) / (67.37 / 100) ** 2,  # ±0.0074
        "H0": 67.37,  # ±0.54
        "Omega_b_h2": 0.02233,  # ±0.00015
        "Omega_k": 0.0,
        "w": -1.0,
        "wa": 0.0,
        "m_nu": 0.06,  # jnp.array([0.06, 0.0, 0.0]),
        "Neff": 3.046,
    },
    # Base-ΛCDM cosmological parameters from Planck
    # TT,TE,EE+lowE+lensing+BAO. Taken from Table 2. in
    # 10.1051/0004-6361/201833910
    # This is Planck18 in astropy
    "PlanckBAO18": {
        "Tcmb": 2.7255,  # from Planck18 arxiv:1807.06209 footnote 14 citing Fixsen 2009
        "Omega_bc": (0.11933 + 0.02242) / (67.66e-2**2),  # ±0.0074
        "H0": 67.66,  # ±0.54
        "Omega_b_h2": 0.02242,  # ±0.00015
        "Omega_k": 0.0,
        "w": -1.0,
        "wa": 0.0,
        "m_nu": 0.06,  # jnp.array([0.06, 0.0, 0.0]),
        "Neff": 3.046,
    },
    # Fiducial cosmology used in DESI 2024 YR1 BAO measurements
    # Referred as abacus_cosm000 at https://abacussummit.readthedocs.io/en/latest/ cosmologies.html
    # Baseline LCDM, Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing mean
    "DESI2024YR1_Fiducial": {
        "Tcmb": 2.7255,  # from Planck18 arxiv:1807.06209 footnote 14 citing Fixsen 2009
        "Omega_bc": (0.02237 + 0.1200) / (67.36 / 100) ** 2,
        "H0": 67.36,  # ±0.54
        "Omega_b_h2": 0.02237,  # ±0.00015
        "Omega_k": 0.0,
        "w": -1.0,
        "wa": 0.0,
        "m_nu": 0.06,  # jnp.array([0.06, 0.0, 0.0]),  # 0.00064420   2.0328
        "Neff": 3.04,
    },
}

# Default fixed parameters for flat w-CDM
CMB_FREE = ["Omega_b_h2", "H0"]
DEFAULT_FREE = {
    "FLCDM": ["Omega_bc"],
    "LCDM": ["Omega_bc", "Omega_k"],
    "FwCDM": ["Omega_bc", "w"],
    "wCDM": ["Omega_bc", "Omega_k", "w"],
    "FwwaCDM": ["Omega_bc", "w", "wa"],
    "wwaCDM": ["Omega_bc", "Omega_k", "w", "wa"],
}

# Default ranges for the exploration of parameters
DEFAULT_RANGE = {
    "Omega_bc": [0.18, 0.48],
    "Omega_k": [-0.1, 0.1],
    "w": [-0.0, -1.5],
    "wa": [-3, 1],
    "Omega_b_h2": [0.01, 0.04],
    "H0": [60.0, 80.0],
}


def lcdm_deviation(**keys):
    """DEPRECATED: Use `get_cosmo_params` instead. This function will be removed in version 1.0.

    This updates Planck18 default data.

    Args:
        **keys: Keyword arguments to update the default parameters.

    Returns:
        dict: The updated parameter dictionary.
    """
    import warnings

    warnings.warn(
        "lcdm_deviation is deprecated and will be removed in version 1.0. Use get_cosmo_params instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    params = AVAILABLE_PARAMETER_SETS["Planck18"].copy()
    params.update(keys)
    return params


def get_cosmo_params(base="PlanckBAO18", **kwargs):
    """Convenience function to retrieve a standard cosmological parameter set.

    Args:
        base: The base set of default parameters.
        **kwargs: Keyword arguments to update the default parameters.

    Returns:
        dict: The updated parameter dictionary.

    Notes:
        Several "base" parameter sets are available such as:
        - Planck18: LCDM best fit to CMB data (Table 1 in Planck coll. VI 2018)
        - PlanckBAO: Same for CMB+BAO (Table 2 in Planck coll. VI 2018)
        - DESI2024YR1_Fiducial: Fiducial cosmology for DESI yr1 analysis
    """
    try:
        params = AVAILABLE_PARAMETER_SETS[base].copy()
    except KeyError:
        raise KeyError(
            f"Unknown parameter set name {base}. Available sets {AVAILABLE_PARAMETER_SETS.keys()}"
        )
    params.update(kwargs)
    return params


def get_constrained_params(priors, cosmology="wwaCDM", base="PlanckBAO18", **kwargs):
    """Convenience funtion to distinguish between parameters constrained by a list of priors and parameters that need to be fixed.
    Args:
        priors (list): list of likelihoods.Chi2 (or derivative) instances
        cosmology (str): Name of the cosmology model to be fitted
        base (str): Name of the reference set of default parameters.
        **kwargs: Deviations to the set of default parameters.

    Returns:
        (dict, dict): (free_params, fixed_params)
    """
    from cosmologix import likelihoods

    fixed = get_cosmo_params(base=base, **kwargs)
    to_free = DEFAULT_FREE[cosmology].copy()
    for p in priors:
        if type(p) is likelihoods.GeometricCMBLikelihood:
            to_free.extend(CMB_FREE)
        elif type(p) is likelihoods.Chi2:
            to_free.append(p.parameter)
        elif type(p) is likelihoods.CalibratedBAOLikelihood:
            to_free.extend(CMB_FREE)
    constrained = {}
    for par in set(to_free):
        constrained[par] = fixed.pop(par)
    return constrained, fixed


def known_priors():
    """Return the list of predifined prior names in the likelihood module"""
    import cosmologix.likelihoods
    import types

    # The convention is that Pre-defined priors in the likelihood
    # module are functions whose name is capitalized, taking no
    # parameters and returning an object of type Chi2
    known_priors = [
        k
        for k in dir(cosmologix.likelihoods)
        if type(getattr(cosmologix.likelihoods, k)) is types.FunctionType
        and k[0].upper() == k[0]
    ]
    return known_priors


def get_prior(p):
    """Retrieves a prior by name from the `cosmologix.likelihoods` module.

    Args:
        p (str): The name of the prior. We make the list case insensitive.

    Returns:
        object: The prior object.
    """
    import cosmologix.likelihoods
    if p.endswith('*'):
        kwarg = {'uncalibrated': True}
        p = p[:-1]
    else:
        kwarg = {}
    original_name = known_priors()
    try:
        name_match = {k.lower(): k for k in original_name}[p.lower()]
    except KeyError:
        raise KeyError(f"Unknown prior name. Available priors: {original_name}")
    return getattr(cosmologix.likelihoods, name_match)(**kwarg)


def get_priors(prior_names, base="PlanckBAO18", **kwargs):
    """Convenience function to build a list of priors

    Args:
        prior_names (list[str]):
            list of predefined prior names (case insensitive)
            see known_priors() for the list of predefined priors.
        base (str): Name of the base cosmology to pick central values from.
        kwargs (dict): specify a set of gaussian priors on parameters as:
                       {param_name: (central_value, precision)}
                      if central value is omitted, it is picked up from the base cosmology.
    Returns:
        list: The resulting list of prior objects

    Example:
        priors = get_priors(['pr4', 'desidr2'], H0=(70, 1))
    """
    import numpy as np
    from cosmologix import likelihoods

    priors = [get_prior(name) for name in prior_names]
    for key, values in kwargs.items():
        if len(values := np.atleast_1d(values)) == 1:
            central_value = get_cosmo_params(base=base)[key]
            precision = values[0]
        else:
            central_value, precision = values
        priors.append(likelihoods.Chi2(key, float(central_value), float(precision)))
    return priors
