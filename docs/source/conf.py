# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# Points Sphinx at the repository root so that autodoc can import the package
# from the source tree rather than from an installed copy.
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyslfp"
copyright = "2025, David Al-Attar, Dan Heathcote"
author = "David Al-Attar, Dan Heathcote"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Generate docs from docstrings.
    "sphinx.ext.napoleon",  # Understand NumPy- and Google-style docstrings.
    "sphinx.ext.viewcode",  # Link from the documentation to the source.
    "sphinx.ext.intersphinx",  # Link types from numpy, scipy and matplotlib.
]

exclude_patterns = []

# The docstrings use Markdown's convention of single backticks for inline code
# (`EarthState`, `regionmask`). Left to itself reStructuredText reads those as
# "title reference" and renders them as italics; this makes them render as code,
# which is what they mean everywhere in this project.
default_role = "code"

# Dataclasses such as EarthModelParameters and LoveNumbers describe their fields
# in an `Attributes:` docstring section *and* declare them as annotated fields.
# Napoleon's default turns that section into standalone `.. attribute::`
# directives, which autodoc then documents a second time from the annotations.
# Rendering them as info-field entries instead keeps the descriptions and leaves
# each field defined exactly once.
napoleon_use_ivar = True

# Types that appear in signatures and return descriptions resolve to the
# upstream documentation; this is worth about seven hundred links across the API
# reference. Two libraries are deliberately absent: pyshtools, whose
# documentation is built with MkDocs and so publishes no objects.inv for Sphinx
# to read, and scipy, whose inventory is large and resolved nothing.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
# An inventory is fetched over the network, so a build can fail for reasons that
# have nothing to do with this repository. The timeout bounds the wait; the
# other half of the answer is that Read the Docs does not build with
# fail_on_warning, so a blip cannot break the published documentation. Only the
# CI docs job treats warnings as errors, where a re-run is a click away.
intersphinx_timeout = 10


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]


# -- Generated API reference -------------------------------------------------
# sphinx-apidoc runs from here rather than from .readthedocs.yaml or a Makefile
# rule, so that a local build, a CI build and a Read the Docs build all produce
# the same pages from the same settings. The .rst files it writes are build
# output, not source, and are gitignored; previously they were committed and
# then hand-edited to add `:no-index:`, which meant every added or renamed
# module needed a manual regeneration step to avoid quietly disappearing from
# the documentation.

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SOURCE_DIR, "..", ".."))


def _run_apidoc(_app):
    """Regenerate the per-module .rst stubs before the build reads them."""
    from sphinx.ext.apidoc import main

    main(
        [
            "--force",
            "--templatedir",
            os.path.join(_REPO_ROOT, "docs", "apidoc_templates"),
            "--output-dir",
            _SOURCE_DIR,
            os.path.join(_REPO_ROOT, "pyslfp"),
        ]
    )


def setup(app):
    app.connect("builder-inited", _run_apidoc)
