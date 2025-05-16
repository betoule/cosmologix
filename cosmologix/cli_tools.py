"""Collection of constants and tools for the command line interface"""

import click
from typer import Option

from cosmologix import parameters

# We defer other imports to improve responsiveness on the command line
# pylint: disable=import-outside-toplevel

# Available priors
AVAILABLE_PRIORS = [
    "Planck2018",
    "PR4",
    "DESIDR1",
    "DESIDR2",
    "DES5yr",
    "Pantheonplus",
    "Union3",
    "SH0ES",
    "BBNNeffSchoneberg2024",
    "BBNSchoneberg2024",
]

PARAM_CHOICES = list(parameters.Planck18.keys()) + ["M", "rd"]


def tuple_list_to_dict(tuple_list):
    """Parse parameters such as --range Omega_bc 0 1 into a dict"""
    result_dict = {}
    for item in tuple_list:
        if len(item) == 2:
            result_dict[item[0]] = item[1]
        else:
            result_dict[item[0]] = list(item[1:])
    return result_dict


def dict_to_list(dictionnary):
    """Convert default dictionnaries to string usable in command line completion"""

    def to_str(v):
        try:
            return " ".join(str(_v) for _v in v)
        except TypeError:
            return str(v)

    def f():

        return [f"{k} {to_str(v)}" for k, v in dictionnary.items()]

    return f


# Option definitions shared between several commands:
COSMOLOGY_OPTION = Option(
    "--cosmology",
    "-c",
    help="Cosmological model",
    show_choices=True,
    autocompletion=lambda: list(parameters.DEFAULT_FREE.keys()),
)
PRIORS_OPTION = Option(
    "--priors",
    "-p",
    help="Priors to use (e.g., Planck18 DESI2024)",
    show_choices=True,
    autocompletion=lambda: AVAILABLE_PRIORS,
)
FIX_OPTION = Option(
    "--fix",
    "-F",
    help="Fix PARAM at VALUE (e.g., -F H0 70)",
    autocompletion=dict_to_list(parameters.Planck18),
    click_type=click.Tuple([str, float]),
)
LABELS_OPTION = Option(
    "--label",
    "-l",
    help="Override labels for contours (e.g., -l 0 DR2)",
    click_type=click.Tuple([int, str]),
)
COLORS_OPTION = Option(
    "--color",
    help="Override color for contours (e.g., --colors 0 red)",
    click_type=click.Tuple([int, str]),
)
FREE_OPTION = Option(
    "--free",
    help="Force release of parameter (e.g., --free Neff)",
    show_choices=True,
    autocompletion=lambda: PARAM_CHOICES,
)
RANGE_OPTION = Option(
    "--range",
    help="Override exploration range for a parameter (e.g., --range Omega_bc 0.1 0.5)",
    show_choices=True,
    autocompletion=dict_to_list(parameters.DEFAULT_RANGE),
    click_type=click.Tuple([str, float, float]),
)
MU_OPTION = Option(
    "--mu",
    help="Distance modulus data file in npy format",
)
MU_COV_OPTION = Option(
    "--mu-cov",
    help="Optional covariance matrix in npy format",
)


def get_prior(p):
    """Retrieve a prior by name"""
    import cosmologix.likelihoods

    return getattr(cosmologix.likelihoods, p)()


def load_mu(mu_file: str, cov_file: str = ""):
    """Load distance measurement."""
    if mu_file is None:
        return []
    import numpy as np
    from cosmologix import likelihoods

    muobs = np.load(mu_file)
    if cov_file:
        cov = np.load(cov_file)
        like = likelihoods.MuMeasurements(muobs["z"], muobs["mu"], cov)
    else:
        like = likelihoods.DiagMuMeasurements(muobs["z"], muobs["mu"], muobs["muerr"])
    return [like]
