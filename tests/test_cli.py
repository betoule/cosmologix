import pytest
import argparse
from unittest.mock import patch
from cosmologix.cli import main
from cosmologix import likelihoods, contours, distances, parameters
import numpy as np


# Test the main entry point
def test_main_cli(tmp_path, capsys):
    best_fit_path = tmp_path / "bestfit.pkl"
    contour_path = tmp_path / "contour.pkl"
    plot_path = tmp_path / "plot.png"
    corner_path = tmp_path / "corner.png"

    # Generate test mu.npy and cov.npy for load_mu
    mu_path = tmp_path / "mu.npy"
    cov_path = tmp_path / "cov.npy"
    # Create synthetic distance modulus data (100 points, mean=40, std=1)
    nz = 100
    z = np.logspace(-2, 0, nz)
    mu_data = distances.mu(parameters.get_cosmo_params(), z) + np.random.normal(
        0, 0.15, size=nz
    )
    # Create a diagonal covariance matrix (variances=0.1)
    cov_data = np.diag(np.full(nz, 0.15**2))
    np.save(mu_path, np.rec.fromarrays([z, mu_data], names=["z", "mu"]))
    np.save(cov_path, cov_data)

    test_cases = [
        f"fit -p DES5yr -A -o {best_fit_path.as_posix()}",
        f"fit -p DES5yr -F H0 70. -F Omega_b_h2 0.02222 -o {best_fit_path.as_posix()}",
        f"fit --mu {mu_path.as_posix()} --mu-cov {cov_path.as_posix()} -F H0 70. -F Omega_b_h2 0.02222 -o {best_fit_path.as_posix()}",
        f"fit -p DES5yr -p PR4 -o {best_fit_path.as_posix()}",
        f"explore Omega_bc w -p Planck2018 -p DES5yr -o {contour_path.as_posix()}",
        f"explore Omega_bc w H0 -p Planck2018 -p DES5yr -o {contour_path.as_posix()}",
        f"contour {contour_path.as_posix()} -o {plot_path.as_posix()}",
        f"corner {contour_path.as_posix()} {best_fit_path.as_posix()} --label 1 'Planck+DES' -o {plot_path.as_posix()}",
    ]
    for test_case in test_cases:
        with patch("sys.argv", ["cosmologix"] + test_case.split()):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0, f"Expected exit code 0 for {test_case}"
        captured = capsys.readouterr()
        assert "saved" in captured.out
