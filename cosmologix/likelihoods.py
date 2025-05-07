"""
Chi squared and log likelihood
"""

from functools import partial
import gzip

import numpy as np
from jax import jit
import jax.numpy as jnp
from . import distances, acoustic_scale, densities, tools


class Chi2:
    """Basic implementation of chi-squared (χ²) evaluation for
    statistical analysis.

    This class provides a framework for computing the chi-squared
    statistic, which is commonly used to evaluate how well a model
    fits a set of observations.  It includes the following methods

    - residuals: Computes the difference between observed data and model predictions.
    - weighted_residuals: Computes residuals normalized by the error.
    - negative_log_likelihood: Computes the sum of squared weighted residuals,
      which corresponds to negative twice the log-likelihood for Gaussian errors.

    In this base implementation we assume that we have a simple
    measurement of a single parameter with its 1-sigma error. The
    class can be derived to implement different behavior by
    overwriting the following attributes:

    - data: The observed data values.
    - model: A function or callable that takes parameters and returns model predictions.
    - error: The uncertainties or standard deviations of the data points.

    """

    def __init__(self, parameter, mean, error):
        """Likelihood for a single parameter measurement

        Parameters:
        - parameter: 'str' name of the measured parameter
        - mean: float measured value
        - error: float 1-sigma error
        """
        self.data = jnp.array([mean])
        self.error = jnp.array([error])
        self.parameter = parameter

    def model(self, params):
        """Model evaluation

        In this basic case we simply return the parameter value
        """
        return jnp.array(params[self.parameter])

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
        """Compute the negative log-likelihood, which is equivalent to half
        the chi-squared statistic for normally distributed errors.

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
        """Draw a Gaussian random realisation of the model

        Used in simulation and test
        """
        self.data = self.model(params) + tools.randn(self.error)


class Chi2FullCov(Chi2):
    """Same as Chi2 but with dense covariane instead of independant errors

    The class assumes that self.upper_factor containts the upper cholesky factor
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
        return self.upper_factor @ self.residuals(params)


class MuMeasurements(Chi2FullCov):
    """Fully correlated measurements of distance modulus at given redshifts

    Note:
    -----

    Using this prior introduce a new nuisance parameter "M" which is
    the absolute magnitude of Supernovae. Be careful when combining
    several supernovae measurement, because they will all share the
    same nuisance parameter.

    """

    def __init__(self, z_cmb, mu, mu_cov=None, weights=None):
        self.z_cmb = jnp.atleast_1d(z_cmb)
        self.data = jnp.atleast_1d(mu)
        if weights is None:
            self.cov = jnp.array(mu_cov)
            self.weights = jnp.linalg.inv(self.cov)
        else:
            self.weights = weights
        self.upper_factor = jnp.linalg.cholesky(self.weights, upper=True)

    def model(self, params):
        return distances.mu(params, self.z_cmb) + params["M"]

    def initial_guess(self, params):
        return dict(params, M=0.0)


class DiagMuMeasurements(Chi2):
    """Independent measurements of distance modulus at given redshifts

    Note:
    -----

    Using this prior introduce a new nuisance parameter "M" which is
    the absolute magnitude of Supernovae. Be careful when combining
    several supernovae measurement, because they will all share the
    same nuisance parameter.
    """

    def __init__(self, z_cmb, mu, mu_err):
        self.z_cmb = jnp.atleast_1d(z_cmb)
        self.data = jnp.atleast_1d(mu)
        self.error = jnp.atleast_1d(mu_err)

    def model(self, params):
        return distances.mu(params, self.z_cmb) + params["M"]

    def initial_guess(self, params):
        return dict(params, M=0.0)


class GeometricCMBLikelihood(Chi2FullCov):
    """An easy-to-work-with summary of CMB measurements

    Note:
    -----

    See e.g. Komatsu et al. 2009 for a discussion on the compression
    of the CMB measurement into a scale measurement. At first order
    the covariance matrix between the density parameters and the
    angular scale capture the same constraints as the scale parameter.
    """

    def __init__(self, mean, covariance, param_names=None):
        """
        Parameters:
        -----------
        mean: best-fit values for the parameters
        covariance: covariance matrix of vector mean
        param_names: Parameter names constrained by the prior
                     (by default ["Omega_bh2", "Omega_c_h2", and "100theta_MC"])
                     Can be any combination of names from the primary parameter vector
                     and secondary parameters computed in the model function
        """
        if param_names is None:
            param_names = ["Omega_b_h2", "Omega_c_h2", "100theta_MC"]
        self.data = jnp.array(mean)
        self.cov = np.array(covariance)
        self.weight_matrix = np.linalg.inv(self.cov)
        self.upper_factor = jnp.array(
            np.linalg.cholesky(self.weight_matrix).T
        )  # , upper=True)
        self.param_names = param_names

    def model(self, params):
        params = densities.process_params(params)
        params["Omega_c_h2"] = params["Omega_c"] * (params["H0"] ** 2 * 1e-4)
        params["Omega_bc_h2"] = params["Omega_bc"] * (params["H0"] ** 2 * 1e-4)
        params["100theta_MC"] = acoustic_scale.theta_MC(params)
        params["theta_MC"] = params["100theta_MC"] / 100.0
        return jnp.array([params[param] for param in self.param_names])
        # return jnp.array([params["Omega_b_h2"], Omega_c_h2, theta_MC(params)])

    def draw(self, params):
        m = self.model(params)
        n = jnp.linalg.solve(self.upper_factor, tools.randn(1, n=len(m)))
        self.data = m + n


class UncalibratedBAOLikelihood(Chi2FullCov):
    """BAO measurements with r_d as a free parameter"""

    def __init__(self, redshifts, data, covariance, dist_type_labels):
        """
        Parameters:
        -----------
        redshifts: BAO redshifts

        data: BAO distances

        covariance: covariance matrix of vector mean

        dist_type_labels: list of labels for distances among
        ['DV_over_rd', 'DM_over_rd', 'DH_over_rd']

        """
        self.redshifts = jnp.asarray(redshifts)
        self.data = jnp.asarray(data)
        self.cov = np.asarray(covariance)
        self.weight_matrix = np.linalg.inv(self.cov)
        self.upper_factor = jnp.array(
            np.linalg.cholesky(self.weight_matrix).T
        )  # , upper=True)
        self.dist_type_labels = dist_type_labels
        if len(self.data) != len(self.dist_type_labels):
            raise ValueError(
                "Distance and dist_type_indices array must have the same length."
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
        choices = [
            distances.dV(params, self.redshifts),
            distances.dM(params, self.redshifts),
            distances.dH(params, self.redshifts),
        ]
        return jnp.choose(self.dist_type_indices, choices, mode="clip") / rd

    def initial_guess(self, params):
        """
        Append relevant starting point for nuisance parameters to the parameter dictionary

        """
        return dict(params, rd=151.0)


class CalibratedBAOLikelihood(UncalibratedBAOLikelihood):
    """BAO measurements with rd computed from other parameters"""

    def model(self, params):
        rd = acoustic_scale.rd_approx(params)
        return super().model(dict(params, rd=rd))

    def initial_guess(self, params):
        """
        Append relevant starting point for nuisance parameters to the parameter dictionary

        """
        return params


@tools.cached
def Pantheonplus():
    """Return likelihood from the Pantheon+SHOES SNe-Ia measurement

    bibcode: 2022ApJ...938..113S
    """
    data = tools.load_csv_from_url(
        "https://github.com/PantheonPlusSH0ES/DataRelease/raw/refs/heads/main/Pantheon+_Data/"
        "4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat",
        delimiter=" ",
    )
    covmat = tools.cached_download(
        "https://github.com/PantheonPlusSH0ES/DataRelease/raw/refs/heads/main/Pantheon+_Data/"
        "4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov"
    )
    cov_matrix = np.loadtxt(covmat)
    nside = int(cov_matrix[0])
    cov_matrix = cov_matrix[1:].reshape((nside, nside))
    np.fill_diagonal(
        cov_matrix, np.diag(cov_matrix)
    )  # + data["MU_SH0ES_ERR_DIAG"] ** 2)
    return MuMeasurements(data["zHD"], data["MU_SH0ES"], cov_matrix)


@tools.cached
def DES5yr():
    """Return likelihood from the DES 5year SNe-Ia survey

    bibcode: 2024ApJ...973L..14D
    """
    des_data = tools.load_csv_from_url(
        "https://github.com/des-science/DES-SN5YR/raw/refs/heads/main/4_DISTANCES_COVMAT/"
        "DES-SN5YR_HD+MetaData.csv"
    )
    covmat = tools.cached_download(
        "https://github.com/des-science/DES-SN5YR/raw/refs/heads/main/4_DISTANCES_COVMAT/"
        "STAT+SYS.txt.gz"
    )
    with gzip.open(covmat, "rt") as f:  # 'rt' mode for text reading
        cov_matrix = np.loadtxt(f)
    nside = int(cov_matrix[0])
    cov_matrix = cov_matrix[1:].reshape((nside, nside))
    np.fill_diagonal(cov_matrix, np.diag(cov_matrix) + des_data["MUERR_FINAL"] ** 2)
    # return DiagMuMeasurements(des_data["zCMB"], des_data["MU"], des_data["MUERR_FINAL"])
    return MuMeasurements(des_data["zHD"], des_data["MU"], cov_matrix)


@tools.cached
def Union3():
    """Return likelihood from the Union 3 compilation

    bibcode: 2023arXiv231112098R
    """
    from astropy.io import fits  # pylint: disable=import-outside-toplevel

    union3_file = tools.cached_download(
        "https://github.com/rubind/union3_release/raw/refs/heads/main/mu_mat_union3_cosmo=2_mu.fits"
    )
    union3_mat = fits.getdata(union3_file)
    z = jnp.array(union3_mat[0, 1:])
    mu = jnp.array(union3_mat[1:, 0])
    inv_cov = jnp.array(union3_mat[1:, 1:])
    return MuMeasurements(z, mu, weights=inv_cov)


@tools.cached
def JLA():
    """Return likelihood from the Joint Light-curve Analysis compilation

    bibcode: 2014A&A...568A..22B
    """
    from astropy.io import fits  # pylint: disable=import-outside-toplevel

    binned_distance_moduli = np.loadtxt(
        tools.cached_download(
            "https://cdsarc.cds.unistra.fr/ftp/J/A+A/568/A22/tablef1.dat"
        )
    )
    cov_mat = fits.getdata(
        tools.cached_download(
            "https://cdsarc.cds.unistra.fr/ftp/J/A+A/568/A22/tablef2.fit"
        )
    )
    return MuMeasurements(
        binned_distance_moduli[:, 0], binned_distance_moduli[:, 1], cov_mat
    )


def Planck2018Prior():
    """Geometric prior for Planck 2018 release

    The values have been extracted from the cosmomc archive. Relevant
    files for the central values and covariance were:

    - base_plikHM_TTTEEE_lowl_lowE.likestats
    - base_plikHM_TTTEEE_lowl_lowE.covmat

    """
    planck2018_prior = GeometricCMBLikelihood(
        [2.2337930e-02, 1.2041740e-01, 1.0409010e00],
        [
            [2.2139987e-08, -1.1786703e-07, 1.6777190e-08],
            [-1.1786703e-07, 1.8664921e-06, -1.4772837e-07],
            [1.6777190e-08, -1.4772837e-07, 9.5788538e-08],
        ],
    )
    return planck2018_prior


def PR4():
    """
    From DESI DR2 results https://arxiv.org/pdf/2503.14738 Appendix A
    """
    return GeometricCMBLikelihood(
        [0.01041, 0.02223, 0.14208],
        jnp.array(
            [
                [0.006621, 0.12444, -1.1929],
                [0.12444, 21.344, -94.001],
                [-1.1929, -94.001, 1488.4],
            ]
        )
        * 1e-9,
        ["theta_MC", "Omega_b_h2", "Omega_bc_h2"],
    )


def DESIDR2Prior(uncalibrated=False):
    """
    From DESI DR2 results https://arxiv.org/pdf/2503.14738 Table IV
    :return:
    """
    Prior = UncalibratedBAOLikelihood if uncalibrated else CalibratedBAOLikelihood
    desi2025_prior = Prior(
        redshifts=[
            0.295,
            0.510,
            0.510,
            0.706,
            0.706,
            0.934,
            0.934,
            1.321,
            1.321,
            1.484,
            1.484,
            2.330,
            2.330,
        ],
        data=[
            7.944,
            13.587,
            21.863,
            17.347,
            19.458,
            21.574,
            17.641,
            27.605,
            14.178,
            30.519,
            12.816,
            38.988,
            8.632,
        ],
        covariance=[
            [0.075**2] + [0] * 12,
            [0, 0.169**2, -0.475 * 0.169 * 0.427] + [0] * 10,
            [0, -0.475 * 0.169 * 0.427, 0.427**2] + [0] * 10,
            [0] * 3 + [0.180**2, -0.423 * 0.180 * 0.332] + [0] * 8,
            [0] * 3 + [-0.423 * 0.180 * 0.332, 0.332**2] + [0] * 8,
            [0] * 5 + [0.153**2, -0.425 * 0.153 * 0.193] + [0] * 6,
            [0] * 5 + [-0.425 * 0.153 * 0.193, 0.193**2] + [0] * 6,
            [0] * 7 + [0.320**2, -0.437 * 0.320 * 0.217] + [0] * 4,
            [0] * 7 + [-0.437 * 0.320 * 0.217, 0.217**2] + [0] * 4,
            [0] * 9 + [0.758**2, -0.489 * 0.758 * 0.513] + [0] * 2,
            [0] * 9 + [-0.489 * 0.758 * 0.513, 0.513**2] + [0] * 2,
            [0] * 11 + [0.531**2, -0.431 * 0.531 * 0.101],
            [0] * 11 + [-0.431 * 0.531 * 0.101, 0.101**2],
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
            "DM_over_rd",
            "DH_over_rd",
            "DM_over_rd",
            "DH_over_rd",
        ],
    )
    return desi2025_prior


def DESIDR1Prior(uncalibrated=False):
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
        data=[
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


class BBNNeffLikelihood(GeometricCMBLikelihood):
    """Prior of the couple (Omega_b_h2, Neff)"""

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

    bbn_prior = Chi2("Omega_b_h2", 0.02218, 0.00055)
    return bbn_prior


def SH0ES():
    """
    H0 measurement from Murakami et al. 2023 (doi:10.1088/1475-7516/2023/11/046)
    """
    return Chi2("H0", 73.29, 0.90)
