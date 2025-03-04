import pytest
import argparse
from unittest.mock import patch
from cosmologix.cli import main
from cosmologix import likelihoods, contours

# Test the main entry point
def test_main_cli(tmp_path, capsys):
    best_fit_path = tmp_path/"bestfit.pkl"
    contour_path = tmp_path/"contour.pkl"
    plot_path = tmp_path/"plot.png"
    test_cases = [
        f"fit -p Planck18 DES-5yr -o {best_fit_path.as_posix()}",
        f"explore Omega_m w -p Planck18 DES-5yr -o {contour_path.as_posix()}",
        f"contour {contour_path.as_posix()} -o {plot_path.as_posix()}",
        ]
    for test_case in test_cases:
        with patch("sys.argv", ["cosmologix"] + test_case.split()):
            main()
        captured = capsys.readouterr()
        assert "saved" in captured.out
