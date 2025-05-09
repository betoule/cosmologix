from cosmologix import likelihoods, contours, tools, display
from cosmologix.parameters import Planck18
from cosmologix.distances import mu
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose


def compare_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if isinstance(dict1[key], (jnp.ndarray, np.ndarray)):
            try:
                assert_allclose(dict1[key], dict2[key])
            except AssertionError:
                return False
        elif isinstance(dict1[key], dict):
            if not compare_dicts(dict1[key], dict2[key]):
                return False
        else:
            if dict1[key] != dict2[key]:
                return False
    return True


# NOTE: The coverage of this test function is a good way to see which
# parts of the codebase are useless
def test_contours(tmp_path):
    fixed = {"Omega_k": 0.0, "m_nu": 0.06, "Neff": 3.046, "Tcmb": 2.7255, "wa": 0.0}
    priors = [likelihoods.Planck2018(), likelihoods.JLA()]
    grid = contours.frequentist_contour_2d_sparse(
        priors, grid={"Omega_bc": [0.18, 0.48, 30], "w": [-0.6, -1.5, 30]}, fixed=fixed
    )
    assert jnp.isfinite(grid["chi2"].any())
    temp_file = tmp_path / "test_file.txt"
    tools.save(grid, temp_file)
    grid2 = tools.load(temp_file)
    assert compare_dicts(grid, grid2)
    display.plot_contours(grid)
    grid_coarse = contours.frequentist_contour_2d(
        priors, grid={"Omega_bc": [0.18, 0.48, 10], "w": [-0.6, -1.5, 10]}, fixed=fixed
    )
    assert jnp.isfinite(grid_coarse["chi2"].any())
    display.plot_contours(temp_file, bestfit=True, filled=True)


if __name__ == "__main__":
    test_contours()
