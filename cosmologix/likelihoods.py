from cosmologix import mu
import jax.numpy as jnp


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


class MuMeasurements(Chi2):
    def __init__(self, z_cmb, mu, mu_err):
        self.z_cmb = jnp.atleast_1d(z_cmb)
        self.data = jnp.atleast_1d(mu)
        self.error = jnp.atleast_1d(mu_err)

    def model(self, params):
        return mu(params, self.z_cmb)


# class GeometricCMBPrior:
#    def __init__(self, mean, covariance):
#        """An easy-to-work-with summary of CMB measurements
#
#        Parameters:
#        -----------
#        mean: best-fit values for Omega_ch2, Omega_bh2, and 100tetha_MC
#
#        covariance: covariance matrix of vector mean
#        """
#        self.mean = jnp.array(mean)
#        self.cov = jnp.array(covariance)
#        self.W = jnp.linalg.inv(self.cov)
#        self.L = jnp.linalg.cholesky(self.W)
#
#    def model(self, params):
#        Omega_c_h2 = Omega_c(params) * (params["H0"] ** 2 * 1e-4)
#        zstar = z_star(params)
#        rsstar = rs(params, zstar)
#        thetastar = rsstar / dM(params, zstar) * 100.0
#        return jnp.array([Omega_c_h2, params["Omega_b_h2"], thetastar])
#
#    def residuals(self, params):
#        return self.mean - self.model(params)
#
#    def weighted_residuals(self, params):
#        return self.L @ self.residuals(params)
#
#    def likelihood(self, params):
#        r = self.weighted_residuals(params)
#        return r.T @ r
#
#
## Extracted from
# planck2018_prior = GeometricCMBPrior(
#    [2.2337930e-02, 1.2041740e-01, 1.0409010e00],
#    [
#        [2.2139987e-08, -1.1786703e-07, 1.6777190e-08],
#        [-1.1786703e-07, 1.8664921e-06, -1.4772837e-07],
#        [1.6777190e-08, -1.4772837e-07, 9.5788538e-08],
#    ],
# )
#


def DES5yr():
    from cosmologix.tools import load_csv_from_url

    des_data = load_csv_from_url(
        "https://github.com/des-science/DES-SN5YR/raw/refs/heads/main/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv"
    )
    return MuMeasurements(des_data["zCMB"], des_data["MU"], des_data["MUERR_FINAL"])
