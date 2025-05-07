"""Best fit cosmologies"""

# Base-ΛCDM cosmological parameters from Planck
# TT,TE,EE+lowE+lensing. Taken from Table 1. in
# 10.1051/0004-6361/201833910
Planck18 = {
    "Tcmb": 2.7255,  # from Planck18 arxiv:1807.06209 footnote 14 citing Fixsen 2009
    "Omega_bc": (0.02233 + 0.1198) / (67.37 / 100) ** 2,  # ±0.0074
    "H0": 67.37,  # ±0.54
    "Omega_b_h2": 0.02233,  # ±0.00015
    "Omega_k": 0.0,
    "w": -1.0,
    "wa": 0.0,
    "m_nu": 0.06,  # jnp.array([0.06, 0.0, 0.0]),
    "Neff": 3.046,
}

# Fiducial cosmology used in DESI 2024 YR1 BAO measurements
# Referred as abacus_cosm000 at https://abacussummit.readthedocs.io/en/latest/ cosmologies.html
# Baseline LCDM, Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing mean
DESI2024YR1_Fiducial = {
    "Tcmb": 2.7255,  # from Planck18 arxiv:1807.06209 footnote 14 citing Fixsen 2009
    "Omega_bc": (0.02237 + 0.1200) / (67.36 / 100) ** 2,
    "H0": 67.36,  # ±0.54
    "Omega_b_h2": 0.02237,  # ±0.00015
    "Omega_k": 0.0,
    "w": -1.0,
    "wa": 0.0,
    "m_nu": 0.06,  # jnp.array([0.06, 0.0, 0.0]),  # 0.00064420   2.0328
    "Neff": 3.04,
}
