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
    "FLCDM": ["Omega_bc"] + CMB_FREE,
    "LCDM": ["Omega_bc", "Omega_k"] + CMB_FREE,
    "FwCDM": ["Omega_bc", "w"] + CMB_FREE,
    "wCDM": ["Omega_bc", "Omega_k", "w"] + CMB_FREE,
    "FwwaCDM": ["Omega_bc", "w", "wa"] + CMB_FREE,
    "wwaCDM": ["Omega_bc", "Omega_k", "w", "wa"] + CMB_FREE,
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
