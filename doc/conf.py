# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../cosmologix"))


import cosmologix

project = "cosmologix"
copyright = "2025, M. Betoule, J. Neveu, D. Kuhn"
author = "M. Betoule, J. Neveu, D. Kuhn"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    #    "sphinx.ext.napoleon",
    #    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    #    "sphinx.ext.mathjax",
    #    "sphinx.ext.intersphinx",
    #    "sphinx_autodoc_typehints",
    #    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
source_suffix = [".rst", "*.md"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinc_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_type_aliases = {
    "Iterable": "Iterable",
    "ArrayLike": "ArrayLike",
}

mathjax3_config = {
    "loader": {"load": ["[tex]/physics"]},
    "tex": {
        "packages": {"[+]": ["physics"]},
    },
}

bibtex_bibfiles = ["paper.bib"]
bibtex_default_style = "unsrt"
