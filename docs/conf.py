# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as get_version

import flucs  # noqa: F401

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "flucs"
copyright = "2026, Plamen Ivanov, Toby Adkins"
author = "Plamen Ivanov, Toby Adkins"

# The full version, including alpha/beta/rc tags
release = get_version(project)
# Major.minor version
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {"ignore-module-all": True}
autodoc_typehints = "description"
autodoc_class_signature = "mixed"

# Numpy-doc config
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_attr_annotations = True
napoleon_preprocess_types = True

# The default role for text marked up `like this`
default_role = "any"

# Tell sphinx what the primary language being documented is.
primary_domain = "py"

# Tell sphinx what the pygments highlight language should be.
highlight_language = "python"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/flucs-code/flucs",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "home_page_in_toc": False,
}

pygments_style = "sphinx"

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
