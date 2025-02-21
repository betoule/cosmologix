from cosmologix.distances import dM, dH, dV
from cosmologix.acoustic_scale import z_star, theta_MC, z_drag, rs
from cosmologix import mu, densities
import jax.numpy as jnp
from cosmologix.tools import randn
from jax import lax, vmap


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


class MuMeasurements(Chi2):
    def __init__(self, z_cmb, mu, mu_err):
        self.z_cmb = jnp.atleast_1d(z_cmb)
        self.data = jnp.atleast_1d(mu)
        self.error = jnp.atleast_1d(mu_err)

    def model(self, params):
        return mu(params, self.z_cmb) + params["M"]

    def initial_guess(self, params):
        return dict(params, M=0.0)


class GeometricCMBLikelihood(Chi2):
    def __init__(self, mean, covariance):
        """An easy-to-work-with summary of CMB measurements

        Parameters:
        -----------
        mean: best-fit values for Omega_bh2, Omega_c_h2, and 100tetha_MC

        covariance: covariance matrix of vector mean
        """
        self.mean = jnp.array(mean)
        self.cov = jnp.array(covariance)
        self.W = jnp.linalg.inv(self.cov)
        self.L = jnp.linalg.cholesky(self.W)

    def model(self, params):
        params = densities.process_params(params)
        Omega_c_h2 = params["Omega_c"] * (params["H0"] ** 2 * 1e-4)

        return jnp.array([params["Omega_b_h2"], Omega_c_h2, theta_MC(params)])

    def residuals(self, params):
        return self.mean - self.model(params)

    def weighted_residuals(self, params):
        return self.L @ self.residuals(params)

    def likelihood(self, params):
        r = self.weighted_residuals(params)
        return r.T @ r

    def draw(self, params):
        m = self.model(params)
        n = jnp.linalg.solve(self.L, randn(1, n=len(m)))
        self.mean = m + n


class UncalibratedBAOLikelihood(Chi2):
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
        self.distances = jnp.asarray(distances)
        self.cov = jnp.asarray(covariance)
        self.W = jnp.linalg.inv(self.cov)
        self.L = jnp.linalg.cholesky(self.W)
        self.dist_type_labels = dist_type_labels
        if len(self.distances) != len(self.dist_type_labels):
            raise ValueError(
                f"Distance and dist_type_indices array must have the same length."
            )
        self.dist_type_indices = self._convert_labels_to_indices()

    def _convert_labels_to_indices(self):
        self.dist_type_indices = [0] * len(self.dist_type_labels)
        for k, label in enumerate(self.dist_type_labels):
            if label == "DV_over_rd":
                self.dist_type_indices[k] = 0
            elif label == "DM_over_rd":
                self.dist_type_indices[k] = 1
            elif label == "DH_over_rd":
                self.dist_type_indices[k] = 2
            else:
                raise ValueError(f"Label {label} not recognized.")
        return jnp.array(self.dist_type_indices)

    def model(self, params) -> jnp.ndarray:
        rd = params['rd']

        def dV_over_rd(z):
            return dV(params, z) / rd

        def dM_over_rd(z):
            return dM(params, z) / rd

        def dH_over_rd(z):
            return dH(params, z) / rd

        branches = (dV_over_rd, dM_over_rd, dH_over_rd)
        zz = jnp.tile(self.redshifts, (len(branches), 1))
        functions = vmap(lambda i, x: lax.switch(i, branches, x))
        dists = functions(jnp.arange(len(branches)), zz)
        return dists

    def residuals(self, params):
        dists = self.model(params)
        dists = dists[self.dist_type_indices, jnp.arange(self.redshifts.size)]
        return self.distances - dists

    def weighted_residuals(self, params):
        return self.L @ self.residuals(params)

    def likelihood(self, params):
        r = self.weighted_residuals(params)
        return r.T @ r
    
    def initial_guess(self, params):
        """
        Append relevant starting point for nuisance parameters to the parameter dictionary

        """
        return dict(params, rd=151.)

class CalibratedBAOLikelihood(UncalibratedBAOLikelihood):
    def model(self, params):
        zdrag = z_drag(params)
        rd = rs(params, zdrag)
        return super().model(dict(params, rd=rd))

    def initial_guess(self, params):
        """
        Append relevant starting point for nuisance parameters to the parameter dictionary

        """
        return params
    
def DES5yr():
    from cosmologix.tools import load_csv_from_url

    des_data = load_csv_from_url(
        "https://github.com/des-science/DES-SN5YR/raw/refs/heads/main/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv"
    )
    return MuMeasurements(des_data["zCMB"], des_data["MU"], des_data["MUERR_FINAL"])


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


# Best fit cosomologies

# Base-ΛCDM cosmological parameters from Planck
# TT,TE,EE+lowE+lensing. Taken from Table 1. in
# 10.1051/0004-6361/201833910
Planck18 = {
    "Tcmb": 2.7255,  # Check this number and report the exact origin
    "Omega_m": (0.02233 + 0.1198) / (67.37 / 100) ** 2,  # ±0.0074
    "H0": 67.37,  # ±0.54
    "Omega_b_h2": 0.02233,  # ±0.00015
    "Omega_k": 0.0,
    "w": -1.0,
    "wa": 0.0,
    "m_nu": 0.06,  # jnp.array([0.06, 0.0, 0.0]),
    "Neff": 3.046,
}
# Planck18 = {
#    "Omega_m": 0.30966,
#    "Tcmb": 2.7255,
#    "Omega_b_h2": 0.0224178,
#    "Omega_k": 0.0,
#    "w": -1.0,
#    "H0": 67.66,
#    "m_nu": 0.06,
#    "Neff": 3.046,
# }
