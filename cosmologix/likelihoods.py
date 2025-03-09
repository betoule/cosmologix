from cosmologix.distances import dM, dH, dV
from cosmologix.acoustic_scale import theta_MC, rd_approx
from cosmologix import mu, densities
import jax.numpy as jnp
from cosmologix.tools import randn
from jax import lax, vmap, jit
from functools import partial
import numpy as np


class Chi2:
    """Abstract implementation of chi-squared (χ²) evaluation for statistical analysis.

    This class provides a framework for computing the chi-squared
    statistic, which is commonly used to evaluate how well a model
    fits a set of observations.  It includes the following methods

    - residuals: Computes the difference between observed data and model predictions.
    - weighted_residuals: Computes residuals normalized by the error.
    - negative_log_likelihood: Computes the sum of squared weighted residuals,
      which corresponds to negative twice the log-likelihood for Gaussian errors.

    It should be derived to additionnally provide the following
    attributes:

    - data: The observed data values.
    - model: A function or callable that takes parameters and returns model predictions.
    - error: The uncertainties or standard deviations of the data points.
    """

    def residuals(self, params):
        """
        Calculate the residuals between data and model predictions.

        Parameters:
        - params: A dictionary or list of model parameters.

        Returns:
        - numpy.ndarray: An array of residuals where residuals = data - model(params).
        """
        return self.data - self.model(params)

    def weighted_residuals(self, params):
        """
        Calculate the weighted residuals, normalizing by the error.

        Parameters:
        - params: A dictionary or list of model parameters.

        Returns:
        - numpy.ndarray: An array where each element is residual/error.
        """
        return self.residuals(params) / self.error

    def negative_log_likelihood(self, params):
        """
        Compute the negative log-likelihood, which is equivalent to half the chi-squared
        statistic for normally distributed errors.

        Parameters:
        - params: A dictionary or list of model parameters.

        Returns:
        - float: The sum of the squares of the weighted residuals, representing
          -2 * ln(likelihood) for Gaussian errors.
        """
        return (self.weighted_residuals(params) ** 2).sum()

    def initial_guess(self, params):
        """
        Append relevant starting point for nuisance parameters to the parameter dictionary

        """
        return params

    def draw(self, params):
        self.data = self.model(params) + randn(self.error)


class Chi2FullCov(Chi2):
    """Same as Chi2 but with dense covariane instead of independant errors

    The class assumes that self.U containts the upper cholesky factor
    of the inverse of the covariance matrix of the measurements.

    """

    def weighted_residuals(self, params):
        """
        Calculate the weighted residuals, normalizing by the error.

        Parameters:
        - params: A dictionary or list of model parameters.

        Returns:
        - numpy.ndarray: An array where each element is residual/error.
        """
        return self.U @ self.residuals(params)


class LikelihoodSum:
    def __init__(self, likelihoods):
        self.likelihoods = likelihoods

    def negative_log_likelihood(self, params):
        return jnp.sum(
            jnp.array([l.negative_log_likelihood(params) for l in self.likelihoods])
        )

    def weighted_residuals(self, params):
        return jnp.hstack([l.weighted_residuals(params) for l in self.likelihoods])

    def initial_guess(self, params):
        for l in self.likelihoods:
            params = l.initial_guess(params)
        return params

    def draw(self, params):
        for l in self.likelihoods:
            l.draw(params)


class MuMeasurements(Chi2FullCov):
    def __init__(self, z_cmb, mu, mu_cov):
        self.z_cmb = jnp.atleast_1d(z_cmb)
        self.data = jnp.atleast_1d(mu)
        self.cov = jnp.array(mu_cov)
        self.weights = jnp.linalg.inv(self.cov)
        self.U = jnp.linalg.cholesky(self.weights, upper=True)

    def model(self, params):
        return mu(params, self.z_cmb) + params["M"]

    def initial_guess(self, params):
        return dict(params, M=0.0)


class DiagMuMeasurements(Chi2):
    def __init__(self, z_cmb, mu, mu_err):
        self.z_cmb = jnp.atleast_1d(z_cmb)
        self.data = jnp.atleast_1d(mu)
        self.error = jnp.atleast_1d(mu_err)

    def model(self, params):
        return mu(params, self.z_cmb) + params["M"]

    def initial_guess(self, params):
        return dict(params, M=0.0)


class GeometricCMBLikelihood(Chi2FullCov):
    def __init__(self, mean, covariance):
        """An easy-to-work-with summary of CMB measurements

        Parameters:
        -----------
        mean: best-fit values for Omega_bh2, Omega_c_h2, and 100tetha_MC

        covariance: covariance matrix of vector mean
        """
        self.data = jnp.array(mean)
        self.cov = jnp.array(covariance)
        self.W = jnp.linalg.inv(self.cov)
        self.U = jnp.linalg.cholesky(self.W, upper=True)

    def model(self, params):
        params = densities.process_params(params)
        Omega_c_h2 = params["Omega_c"] * (params["H0"] ** 2 * 1e-4)

        return jnp.array([params["Omega_b_h2"], Omega_c_h2, theta_MC(params)])

    def draw(self, params):
        m = self.model(params)
        n = jnp.linalg.solve(self.U, randn(1, n=len(m)))
        self.data = m + n


class UncalibratedBAOLikelihood(Chi2FullCov):
    def __init__(self, redshifts, distances, covariance, dist_type_labels):
        """An easy-to-work-with summary of CMB measurements

        Parameters:
        -----------
        redshifts: BAO redshifts

        distances: BAO distances

        covariance: covariance matrix of vector mean

        dist_type_labels: list of labels for distances among ['DV_over_rd', 'DM_over_rd', 'DH_over_rd']
        """
        self.redshifts = jnp.asarray(redshifts)
        self.data = jnp.asarray(distances)
        self.cov = jnp.asarray(covariance)
        self.W = jnp.linalg.inv(self.cov)
        self.U = jnp.linalg.cholesky(self.W, upper=True)
        self.dist_type_labels = dist_type_labels
        if len(self.data) != len(self.dist_type_labels):
            raise ValueError(
                f"Distance and dist_type_indices array must have the same length."
            )
        self.dist_type_indices = self._convert_labels_to_indices()

    def _convert_labels_to_indices(self):
        label_map = {
            "DV_over_rd": 0,
            "DM_over_rd": 1,
            "DH_over_rd": 2,
        }
        return np.array([label_map[label] for label in self.dist_type_labels])

    @partial(jit, static_argnums=(0,))
    def model(self, params) -> jnp.ndarray:
        rd = params["rd"]
        _dV = dV(params, self.redshifts)
        _dM = dM(params, self.redshifts)
        _dH = dH(params, self.redshifts)
        return jnp.choose(self.dist_type_indices, [_dV, _dM, _dH], mode="clip") / rd

    def initial_guess(self, params):
        """
        Append relevant starting point for nuisance parameters to the parameter dictionary

        """
        return dict(params, rd=151.0)


class CalibratedBAOLikelihood(UncalibratedBAOLikelihood):
    def model(self, params):
        rd = rd_approx(params)
        return super().model(dict(params, rd=rd))

    def initial_guess(self, params):
        """
        Append relevant starting point for nuisance parameters to the parameter dictionary

        """
        return params


def Pantheonplus():
    from cosmologix.tools import load_csv_from_url, cached_download
    import gzip

    data = load_csv_from_url(
        "https://github.com/PantheonPlusSH0ES/DataRelease/raw/refs/heads/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat",
        delimiter=" ",
    )
    covmat = cached_download(
        "https://github.com/PantheonPlusSH0ES/DataRelease/raw/refs/heads/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov"
    )
    cov_matrix = np.loadtxt(covmat)
    nside = int(cov_matrix[0])
    cov_matrix = cov_matrix[1:].reshape((nside, nside))
    np.fill_diagonal(
        cov_matrix, np.diag(cov_matrix)
    )  # + data["MU_SH0ES_ERR_DIAG"] ** 2)
    return MuMeasurements(data["zHD"], data["MU_SH0ES"], cov_matrix)


def DES5yr():
    from cosmologix.tools import load_csv_from_url, cached_download
    import gzip

    des_data = load_csv_from_url(
        "https://github.com/des-science/DES-SN5YR/raw/refs/heads/main/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv"
    )
    covmat = cached_download(
        "https://github.com/des-science/DES-SN5YR/raw/refs/heads/main/4_DISTANCES_COVMAT/STAT+SYS.txt.gz"
    )
    with gzip.open(covmat, "rt") as f:  # 'rt' mode for text reading
        cov_matrix = np.loadtxt(f)
    nside = int(cov_matrix[0])
    cov_matrix = cov_matrix[1:].reshape((nside, nside))
    np.fill_diagonal(cov_matrix, np.diag(cov_matrix) + des_data["MUERR_FINAL"] ** 2)
    # return DiagMuMeasurements(des_data["zCMB"], des_data["MU"], des_data["MUERR_FINAL"])
    return MuMeasurements(des_data["zHD"], des_data["MU"], cov_matrix)


def JLA():
    from cosmologix.tools import cached_download
    from astropy.io import fits

    binned_distance_moduli = np.loadtxt(
        cached_download("https://cdsarc.cds.unistra.fr/ftp/J/A+A/568/A22/tablef1.dat")
    )
    cov_mat = fits.getdata(
        cached_download("https://cdsarc.cds.unistra.fr/ftp/J/A+A/568/A22/tablef2.fit")
    )
    return MuMeasurements(
        binned_distance_moduli[:, 0], binned_distance_moduli[:, 1], cov_mat
    )


# Extracted from
def Planck2018Prior():
    planck2018_prior = GeometricCMBLikelihood(
        [2.2337930e-02, 1.2041740e-01, 1.0409010e00],
        [
            [2.2139987e-08, -1.1786703e-07, 1.6777190e-08],
            [-1.1786703e-07, 1.8664921e-06, -1.4772837e-07],
            [1.6777190e-08, -1.4772837e-07, 9.5788538e-08],
        ],
    )
    return planck2018_prior


def DESI2024Prior(uncalibrated=False):
    """
    From DESI YR1 results https://arxiv.org/pdf/2404.03002 Table 1
    :return:
    """
    Prior = UncalibratedBAOLikelihood if uncalibrated else CalibratedBAOLikelihood
    desi2024_prior = Prior(
        redshifts=[
            0.295,
            0.510,
            0.510,
            0.706,
            0.706,
            0.930,
            0.930,
            1.317,
            1.317,
            1.491,
            2.330,
            2.330,
        ],
        distances=[
            7.93,
            13.62,
            20.98,
            16.85,
            20.08,
            21.71,
            17.88,
            27.79,
            13.82,
            26.07,
            39.71,
            8.52,
        ],
        covariance=[
            [0.15**2] + [0] * 11,
            [0, 0.25**2, -0.445 * 0.25 * 0.61] + [0] * 9,
            [0, -0.445 * 0.25 * 0.61, 0.61**2] + [0] * 9,
            [0] * 3 + [0.32**2, -0.420 * 0.32 * 0.60] + [0] * 7,
            [0] * 3 + [-0.420 * 0.32 * 0.60, 0.60**2] + [0] * 7,
            [0] * 5 + [0.28**2, -0.389 * 0.28 * 0.35] + [0] * 5,
            [0] * 5 + [-0.389 * 0.28 * 0.35, 0.35**2] + [0] * 5,
            [0] * 7 + [0.69**2, -0.444 * 0.69 * 0.42] + [0] * 3,
            [0] * 7 + [-0.444 * 0.69 * 0.42, 0.42**2] + [0] * 3,
            [0] * 9 + [0.67**2] + [0] * 2,
            [0] * 10 + [0.94**2, -0.477 * 0.94 * 0.17],
            [0] * 10 + [-0.477 * 0.94 * 0.17, 0.17**2],
        ],
        dist_type_labels=[
            "DV_over_rd",
            "DM_over_rd",
            "DH_over_rd",
            "DM_over_rd",
            "DH_over_rd",
            "DM_over_rd",
            "DH_over_rd",
            "DM_over_rd",
            "DH_over_rd",
            "DV_over_rd",
            "DM_over_rd",
            "DH_over_rd",
        ],
    )
    return desi2024_prior


class BBNLikelihood(Chi2):
    """
    BBN measurement from https://arxiv.org/abs/2401.15054
    """

    def __init__(self, omega_b_h2, omega_b_h2_err):
        self.data = jnp.asarray([omega_b_h2])
        self.error = jnp.asarray([omega_b_h2_err])

    def model(self, params):
        return jnp.array([params["Omega_b_h2"]])


class BBNNeffLikelihood(GeometricCMBLikelihood):

    def __init__(self, mean, covariance):
        GeometricCMBLikelihood.__init__(self, mean, covariance)

    def model(self, params):
        return jnp.array([params["Omega_b_h2"], params["Neff"]])


def BBNNeffSchoneberg2024Prior():
    """
    BBN measurement from https://arxiv.org/abs/2401.15054
    """

    bbn_prior = BBNNeffLikelihood(
        [0.02196, 3.034],
        [[4.03112260e-07, 7.30390042e-05], [7.30390042e-05, 4.52831584e-02]],
    )
    return bbn_prior


def BBNSchoneberg2024Prior():
    """
    BBN measurement from https://arxiv.org/abs/2401.15054
    """

    bbn_prior = BBNLikelihood(0.02218, 0.00055)
    return bbn_prior


#######################
# Best fit cosmologies
#######################

# Base-ΛCDM cosmological parameters from Planck
# TT,TE,EE+lowE+lensing. Taken from Table 1. in
# 10.1051/0004-6361/201833910
Planck18 = {
    "Tcmb": 2.7255,  # from Planck18 arxiv:1807.06209 footnote 14 citing Fixsen 2009
    "Omega_m": (0.02233 + 0.1198) / (67.37 / 100) ** 2,  # ±0.0074
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
    "Omega_m": (0.02237 + 0.1200) / (67.36 / 100) ** 2,
    "H0": 67.36,  # ±0.54
    "Omega_b_h2": 0.02237,  # ±0.00015
    "Omega_k": 0.0,
    "w": -1.0,
    "wa": 0.0,
    "m_nu": 0.06,  # jnp.array([0.06, 0.0, 0.0]),  # 0.00064420   2.0328
    "Neff": 3.04,
}
