"""Collection of constants and tools for the command line interface."""

from typing import Optional

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

PARAM_CHOICES = list(parameters.get_cosmo_params().keys()) + ["M", "rd"]


def tuple_list_to_dict(tuple_list):
    """Parses a list of tuples into a dictionary.

    For example, `[('Omega_bc', 0, 1)]` becomes `{'Omega_bc': [0, 1]}`.

    Args:
        tuple_list (list): A list of tuples.

    Returns:
        dict: The parsed dictionary.
    """
    result_dict = {}
    for item in tuple_list:
        if len(item) == 2:
            result_dict[item[0]] = item[1]
        else:
            result_dict[item[0]] = list(item[1:])
    return result_dict


def dict_to_list(dictionnary):
    """Converts a dictionary to a list of strings for command-line completion.

    Args:
        dictionnary (dict): The dictionary to convert.

    Returns:
        list: A list of strings.
    """

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
    autocompletion=dict_to_list(parameters.get_cosmo_params()),
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
    """Retrieves a prior by name from the `cosmologix.likelihoods` module.

    Args:
        p (str): The name of the prior.

    Returns:
        object: The prior object.
    """
    import cosmologix.likelihoods

    return getattr(cosmologix.likelihoods, p)()


def permissive_load(name):
    """Loads a NumPy file if the input is a string.

    If the input is not a string, it is assumed to be already loaded and is
    returned directly.

    Args:
        name (str or object): The name of the file to load or an already
            loaded object.

    Returns:
        object: The loaded object.
    """
    import numpy as np

    if isinstance(name, str):
        return np.load(name)
    return name


def load_mu(mu_file: str, cov_file: Optional[str] = None):
    """Loads distance modulus measurements.

    Args:
        mu_file (str): The path to the distance modulus data file.
        cov_file (str, optional): The path to the covariance matrix file.
            Defaults to None.

    Returns:
        list: A list containing the likelihood object.
    """
    if mu_file is None:
        return []
    from cosmologix import likelihoods

    muobs = permissive_load(mu_file)
    if cov_file is not None:
        cov = permissive_load(cov_file)
        like = likelihoods.MuMeasurements(muobs["z"], muobs["mu"], cov)
    else:
        like = likelihoods.DiagMuMeasurements(muobs["z"], muobs["mu"], muobs["muerr"])
    return [like]
