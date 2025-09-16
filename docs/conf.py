# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import types
import subprocess

sys.path.insert(0, os.path.abspath(".") + "/../src/python/")

project = "Spyker"
copyright = "2021-2025, Shahriar Rezghi"
author = "Shahriar Rezghi"
release = "2025"

subprocess.check_call(["doxygen", "Doxyfile"], cwd=os.path.dirname(__file__))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["breathe", "sphinx_rtd_theme", "sphinx_togglebutton", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

breathe_projects = {"Spyker": "_build/xml/"}
breathe_default_project = "Spyker"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo-64.png"
