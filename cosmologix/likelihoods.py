"""Chi squared and log likelihood"""

from functools import partial
import gzip

import numpy as np
from jax import jit
import jax.numpy as jnp
from . import distances, acoustic_scale, densities, tools


class Chi2:
    """Basic implementation of chi-squared (χ²) evaluation.

    This class provides a framework for computing the chi-squared statistic,
    which is commonly used to evaluate how well a model fits a set of
    observations.

    Attributes:
        data: The observed data values.
        model: A function that takes parameters and returns model predictions.
        error: The uncertainties of the data points.
    """

    def __init__(self, parameter, mean, error):
        """Initializes the Chi2 class for a single parameter measurement.

        Args:
            parameter (str): The name of the measured parameter.
            mean (float): The measured value.
            error (float): The 1-sigma error.
        """
        self.data = jnp.array([mean])
        self.error = jnp.array([error])
        self.parameter = parameter

    def model(self, params):
        """Evaluates the model.

        In this basic case, it simply returns the parameter value.

        Args:
            params (dict): A dictionary of model parameters.

        Returns:
            jnp.ndarray: The model prediction.
        """
        return jnp.array(params[self.parameter])

    def residuals(self, params):
        """Calculates the residuals between data and model predictions.

        Args:
            params (dict): A dictionary of model parameters.

        Returns:
            jnp.ndarray: An array of residuals (data - model).
        """
        return self.data - self.model(params)

    def weighted_residuals(self, params):
        """Calculates the weighted residuals, normalizing by the error.

        Args:
            params (dict): A dictionary of model parameters.

        Returns:
            jnp.ndarray: An array where each element is residual/error.
        """
        return self.residuals(params) / self.error

    def negative_log_likelihood(self, params):
        """Computes the negative log-likelihood.

        This is equivalent to half the chi-squared statistic for normally
        distributed errors.

        Args:
            params (dict): A dictionary of model parameters.

        Returns:
            float: The sum of the squares of the weighted residuals.
        """
        return (self.weighted_residuals(params) ** 2).sum()

    def initial_guess(self, params):
        """Appends starting points for nuisance parameters.

        Args:
            params (dict): The initial parameter dictionary.

        Returns:
            dict: The updated parameter dictionary.
        """
        return params

    def draw(self, params):
        """Draws a Gaussian random realization of the model.

        Used in simulations and tests.

        Args:
            params (dict): A dictionary of model parameters.
        """
        self.data = self.model(params) + tools.randn(self.error)


class Chi2FullCov(Chi2):
    """Chi2 evaluation with a dense covariance matrix.

    This class assumes that `self.upper_factor` contains the upper Cholesky
    factor of the inverse of the covariance matrix of the measurements.
    """

    def weighted_residuals(self, params):
        """Calculates the weighted residuals using the covariance matrix.

        Args:
            params (dict): A dictionary of model parameters.

        Returns:
            jnp.ndarray: The array of weighted residuals.
        """
        return self.upper_factor @ self.residuals(params)


class MuMeasurements(Chi2FullCov):
    """Likelihood for fully correlated measurements of distance modulus.

    Note:
        Using this prior introduces a new nuisance parameter "M", the absolute
        magnitude of Supernovae. Be careful when combining several supernovae
        measurements, as they will all share the same nuisance parameter.
    """

    def __init__(self, z_cmb, mu, mu_cov=None, weights=None):
        """Initializes the MuMeasurements class.

        Args:
            z_cmb (jnp.ndarray): Redshifts of the measurements.
            mu (jnp.ndarray): Distance moduli.
            mu_cov (jnp.ndarray, optional): Covariance matrix of the distance
                moduli. Defaults to None.
            weights (jnp.ndarray, optional): Inverse of the covariance matrix.
                If not provided, it is computed from `mu_cov`. Defaults to None.
        """
        self.z_cmb = jnp.atleast_1d(z_cmb)
        self.data = jnp.atleast_1d(mu)
        if weights is None:
            self.cov = jnp.array(mu_cov)
            self.weights = jnp.linalg.inv(self.cov)
        else:
            self.weights = weights
        self.upper_factor = jnp.linalg.cholesky(self.weights, upper=True)

    def model(self, params):
        """Calculates the model for the distance modulus.

        Args:
            params (dict): A dictionary of model parameters.

        Returns:
            jnp.ndarray: The predicted distance moduli.
        """
        return distances.mu(params, self.z_cmb) + params["M"]

    def initial_guess(self, params):
        """Appends starting points for nuisance parameters.

        Args:
            params (dict): The initial parameter dictionary.

        Returns:
            dict: The updated parameter dictionary.
        """
        return dict(params, M=0.0)


class DiagMuMeasurements(Chi2):
    """Likelihood for independent measurements of distance modulus.

    Note:
        Using this prior introduces a new nuisance parameter "M", the absolute
        magnitude of Supernovae. Be careful when combining several supernovae
        measurements, as they will all share the same nuisance parameter.
    """

    def __init__(self, z_cmb, mu, mu_err):
        """Initializes the DiagMuMeasurements class.

        Args:
            z_cmb (jnp.ndarray): Redshifts of the measurements.
            mu (jnp.ndarray): Distance moduli.
            mu_err (jnp.ndarray): Errors on the distance moduli.
        """
        self.z_cmb = jnp.atleast_1d(z_cmb)
        self.data = jnp.atleast_1d(mu)
        self.error = jnp.atleast_1d(mu_err)

    def model(self, params):
        """Calculates the model for the distance modulus.

        Args:
            params (dict): A dictionary of model parameters.

        Returns:
            jnp.ndarray: The predicted distance moduli.
        """
        return distances.mu(params, self.z_cmb) + params["M"]

    def initial_guess(self, params):
        """Appends starting points for nuisance parameters.

        Args:
            params (dict): The initial parameter dictionary.

        Returns:
            dict: The updated parameter dictionary.
        """
        return dict(params, M=0.0)


class GeometricCMBLikelihood(Chi2FullCov):
    """A compressed summary of CMB measurements.

    Note:
        See e.g. Komatsu et al. 2009 for a discussion on the compression of
        the CMB measurement into a scale measurement. At first order, the
        covariance matrix between the density parameters and the angular scale
        captures the same constraints as the scale parameter.
    """

    def __init__(self, mean, covariance, param_names=None):
        """Initializes the GeometricCMBLikelihood class.

        Args:
            mean (list or jnp.ndarray): Best-fit values for the parameters.
            covariance (list or jnp.ndarray): Covariance matrix of the mean vector.
            param_names (list, optional): Parameter names constrained by the
                prior. Defaults to `["Omega_b_h2", "Omega_c_h2", "100theta_MC"]`.
                Can be any combination of names from the primary parameter
                vector and secondary parameters computed in the model function.
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
        """Calculates the model for the CMB parameters.

        Args:
            params (dict): A dictionary of model parameters.

        Returns:
            jnp.ndarray: The predicted CMB parameters.
        """
        params = densities.process_params(params)
        params["Omega_c_h2"] = params["Omega_c"] * (params["H0"] ** 2 * 1e-4)
        params["Omega_bc_h2"] = params["Omega_bc"] * (params["H0"] ** 2 * 1e-4)
        params["100theta_MC"] = acoustic_scale.theta_MC(params)
        params["theta_MC"] = params["100theta_MC"] / 100.0
        return jnp.array([params[param] for param in self.param_names])
        # return jnp.array([params["Omega_b_h2"], Omega_c_h2, theta_MC(params)])

    def draw(self, params):
        """Draws a random realization of the model.

        Args:
            params (dict): A dictionary of model parameters.
        """
        m = self.model(params)
        n = jnp.linalg.solve(self.upper_factor, tools.randn(1, n=len(m)))
        self.data = m + n


class UncalibratedBAOLikelihood(Chi2FullCov):
    """Likelihood for BAO measurements with r_d as a free parameter."""

    def __init__(self, redshifts, data, covariance, dist_type_labels):
        """Initializes the UncalibratedBAOLikelihood class.

        Args:
            redshifts (list or jnp.ndarray): BAO redshifts.
            data (list or jnp.ndarray): BAO distances.
            covariance (list or jnp.ndarray): Covariance matrix of the mean vector.
            dist_type_labels (list): List of labels for distances among
                ['DV_over_rd', 'DM_over_rd', 'DH_over_rd'].
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
        """Converts distance type labels to indices."""
        label_map = {
            "DV_over_rd": 0,
            "DM_over_rd": 1,
            "DH_over_rd": 2,
        }
        return np.array([label_map[label] for label in self.dist_type_labels])

    @partial(jit, static_argnums=(0,))
    def model(self, params) -> jnp.ndarray:
        """Calculates the model for the BAO distances.

        Args:
            params (dict): A dictionary of model parameters.

        Returns:
            jnp.ndarray: The predicted BAO distances.
        """
        rd = params["rd"]
        choices = [
            distances.dV(params, self.redshifts),
            distances.dM(params, self.redshifts),
            distances.dH(params, self.redshifts),
        ]
        return jnp.choose(self.dist_type_indices, choices, mode="clip") / rd

    def initial_guess(self, params):
        """Appends starting points for nuisance parameters."""
        return dict(params, rd=151.0)


class CalibratedBAOLikelihood(UncalibratedBAOLikelihood):
    """Likelihood for BAO measurements with rd computed from other parameters."""

    def model(self, params):
        """Calculates the model for the BAO distances.

        Args:
            params (dict): A dictionary of model parameters.

        Returns:
            jnp.ndarray: The predicted BAO distances.
        """
        rd = acoustic_scale.rd_approx(params)
        return super().model(dict(params, rd=rd))

    def initial_guess(self, params):
        """Appends starting points for nuisance parameters."""
        return params


@tools.cached
def Pantheonplus():
    """Returns the likelihood from the Pantheon+SHOES SNe-Ia measurement.

    See bibcode: 2022ApJ...938..113S.
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
    """Returns the likelihood from the DES 5-year SNe-Ia survey.

    See bibcode: 2024ApJ...973L..14D.
    """
    des_data = tools.load_csv_from_url(
        "https://raw.githubusercontent.com/des-science/DES-SN5YR/refs/tags/v1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD%2BMetaData.csv"
    )
    covmat = tools.cached_download(
        "https://github.com/des-science/DES-SN5YR/raw/refs/tags/v1.2/4_DISTANCES_COVMAT/STAT+SYS.txt.gz"
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
    """Returns the likelihood from the Union 3 compilation.

    See bibcode: 2023arXiv231112098R.
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
    """Returns the likelihood from the Joint Light-curve Analysis compilation.

    See bibcode: 2014A&A...568A..22B.
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


def Planck2018():
    """Returns the geometric prior from the Planck 2018 release.

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
    """Returns the geometric prior from DESI DR2 results.

    See https://arxiv.org/pdf/2503.14738 Appendix A.
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


def DESIDR2(uncalibrated=False):
    """Returns the likelihood from DESI DR2 results.

    See https://arxiv.org/pdf/2503.14738 Table IV.
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


def DESIDR1(uncalibrated=False):
    """Returns the likelihood from DESI YR1 results.

    See https://arxiv.org/pdf/2404.03002 Table 1.
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
    """Prior on the couple (Omega_b_h2, Neff)."""

    def __init__(self, mean, covariance):
        """Initializes the BBNNeffLikelihood class.

        Args:
            mean (list or jnp.ndarray): Best-fit values for the parameters.
            covariance (list or jnp.ndarray): Covariance matrix of the mean vector.
        """
        GeometricCMBLikelihood.__init__(self, mean, covariance)

    def model(self, params):
        """Calculates the model for the BBN parameters.

        Args:
            params (dict): A dictionary of model parameters.

        Returns:
            jnp.ndarray: The predicted BBN parameters.
        """
        return jnp.array([params["Omega_b_h2"], params["Neff"]])


def BBNNeffSchoneberg2024():
    """Returns the BBN measurement from Schoeneberg et al. 2024.

    See https://arxiv.org/abs/2401.15054.
    """

    bbn_prior = BBNNeffLikelihood(
        [0.02196, 3.034],
        [[4.03112260e-07, 7.30390042e-05], [7.30390042e-05, 4.52831584e-02]],
    )
    return bbn_prior


def BBNSchoneberg2024():
    """Returns the BBN measurement from Schoeneberg et al. 2024.

    See https://arxiv.org/abs/2401.15054.
    """

    bbn_prior = Chi2("Omega_b_h2", 0.02218, 0.00055)
    return bbn_prior


def SH0ES():
    """Returns the H0 measurement from Murakami et al. 2023.

    See doi:10.1088/1475-7516/2023/11/046.
    """
    return Chi2("H0", 73.29, 0.90)
