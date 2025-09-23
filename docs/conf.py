# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
import biomol

sys.path.insert(0, os.path.abspath("../src"))


project = "BioMol"
copyright = "2025, Lee Howon"
author = "Lee Howon"
release = biomol.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_favicon",
    "numpydoc",
    "myst_nb",
]


# autosummary / autodoc
autosummary_generate = True
autoclass_content = "class"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_typehints_format = "short"
add_module_names = False


# numpydoc
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = "BioMol"
html_favicon = "_static/favicon-32x32.png"
html_static_path = ["_static"]

html_theme_options = {
    "show_nav_level": 1,
    "navigation_depth": 2,
    "logo": {
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
    },
}

# favicons
# reference: https://github.com/pydata/pydata-sphinx-theme/blob/main/docs/conf.py
favicons = [
    # generic icons compatible with most browsers
    "favicon-32x32.png",
    "favicon-16x16.png",
    # chrome specific
    "android-chrome-192x192.png",
    "android-chrome-512x512.png",
    # apple icons
    {"rel": "apple-touch-icon", "href": "apple-touch-icon.png"},
]
