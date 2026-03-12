# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# This line points Sphinx to the root directory of your project so it can find your library.
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyslfp"
copyright = "2025, David Al-Attar, Dan Heathcote"
author = "David Al-Attar, Dan Heathcote"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add the Sphinx extensions necessary for a modern documentation site.
extensions = [
    "sphinx.ext.autodoc",  # Automatically generate docs from docstrings.
    "sphinx.ext.napoleon",  # Enables Sphinx to understand NumPy-style docstrings.
    "sphinx.ext.viewcode",  # Adds links to the source code from the documentation.
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set the HTML theme to 'furo' for a clean, modern look.
html_theme = "furo"
html_static_path = ["_static"]
