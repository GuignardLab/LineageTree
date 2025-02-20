# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../src/LineageTree"))

project = "lineagetree"
copyright = "2025, Leo Guignard"  # noqa: A001
author = "Leo Guignard"
release = "1.7.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # If using Google/NumPy docstrings
    "sphinx.ext.viewcode",  # Links to source code
    "myst_parser",  # Markdown support
    "sphinx.ext.mathjax",  # MathJax support
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

autodoc_inherit_docstrings = True

html_theme_options = {
    "repository_url": "https://github.com/Guignardlab/lineagetree",
    "use_repository_button": True,
    "path_to_docs": "docs",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "deepnote_url": "https://deepnote.com/",
        "notebook_interface": "jupyterlab",
        "thebe": True,
    },
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Guignardlab/lineagetree",
            "icon": "fa-brands fa-github",
        },
    ],
}
