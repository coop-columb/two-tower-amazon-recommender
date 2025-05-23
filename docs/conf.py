"""Configuration file for the Sphinx documentation builder."""

# -- Project information -----------------------------------------------------
project = "Two-Tower Amazon Recommender"
copyright = "2025, Alexander Cooperstone"
author = "Alexander Cooperstone"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------
# MyST parser configuration
myst_enable_extensions = [
    "deflist",
    "tasklist",
    "html_image",
]
