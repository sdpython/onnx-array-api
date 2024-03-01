import os
import sys
from sphinx_runpython.github_link import make_linkcode_resolve
from sphinx_runpython.conf_helper import has_dvipng, has_dvisvgm
from onnx_array_api import __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_runpython.epkg",
    "sphinx_runpython.gdot",
    "sphinx_runpython.runpython",
]

if has_dvisvgm():
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
elif has_dvipng():
    extensions.append("sphinx.ext.pngmath")
    imgmath_image_format = "png"
else:
    extensions.append("sphinx.ext.mathjax")

templates_path = ["_templates"]
html_logo = "_static/logo.png"
source_suffix = ".rst"
master_doc = "index"
project = "onnx-array-api"
copyright = "2023-2024, Xavier Dupré"
author = "Xavier Dupré"
version = __version__
release = __version__
language = "en"
exclude_patterns = []
pygments_style = "sphinx"
todo_include_todos = True

html_theme = "furo"
html_theme_path = ["_static"]
html_theme_options = {}
html_static_path = ["_static"]
html_sourcelink_suffix = ""

issues_github_path = "sdpython/onnx-array-api"

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "onnx-array-api",
    (
        "https://github.com/sdpython/onnx-array-api/"
        "blob/{revision}/{package}/"
        "{path}#L{lineno}"
    ),
)

latex_elements = {
    "papersize": "a4",
    "pointsize": "10pt",
    "title": project,
}

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "onnx": ("https://onnx.ai/onnx/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "sklearn-onnx": ("https://onnx.ai/sklearn-onnx/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Check intersphinx reference targets exist
nitpicky = True
# See also scikit-learn/scikit-learn#26761
nitpick_ignore = [
    ("py:class", "False"),
    ("py:class", "True"),
    ("py:class", "pipeline.Pipeline"),
    ("py:class", "default=sklearn.utils.metadata_routing.UNCHANGED"),
]

nitpick_ignore_regex = [
    ("py:func", ".*numpy[.].*"),
    ("py:func", ".*scipy[.].*"),
    ("py:class", ".*onnxruntime[.].*"),
    ("py:class", ".*onnx_array_api.npx.npx_types.OptParTypeTupleType_.*"),
    ("py:class", ".*onnx_array_api.npx.npx_types.ParType[a-z].*"),
    ("py:class", ".*onnx_array_api.npx.npx_types.OptTensorType_.*"),
    ("py:class", ".*onnx_array_api.npx.npx_types.TensorType_.*"),
    ("py:class", ".*onnx_array_api.npx.npx_types.[ui].*"),
]

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": os.path.join(os.path.dirname(__file__), "examples"),
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
}

epkg_dictionary = {
    "Array API": "https://data-apis.org/array-api/",
    "ArrayAPI": (
        "https://data-apis.org/array-api/",
        ("2022.12/API_specification/generated/array_api.{0}.html", 1),
    ),
    "ast": "https://docs.python.org/3/library/ast.html",
    "cProfile.Profile": "https://docs.python.org/3/library/profile.html#profile.Profile",
    "DOT": "https://graphviz.org/doc/info/lang.html",
    "Graphviz": "https://graphviz.org/",
    "inner API": "https://onnx.ai/onnx/intro/python.html",
    "JIT": "https://en.wikipedia.org/wiki/Just-in-time_compilation",
    "onnx": "https://onnx.ai/onnx/",
    "onnx-graphsurgeon": "https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html",
    "onnx.helper": "https://onnx.ai/onnx/api/helper.html",
    "ONNX": "https://onnx.ai/",
    "ONNX Operators": "https://onnx.ai/onnx/operators/",
    "onnxruntime": "https://onnxruntime.ai/",
    "onnxruntime-training": "https://onnxruntime.ai/docs/get-started/training-on-device.html",
    "numpy": "https://numpy.org/",
    "numba": "https://numba.pydata.org/",
    "onnx-array-api": ("https://sdpython.github.io/doc/onnx-array-api/dev/"),
    "onnxscript": "https://github.com/microsoft/onnxscript",
    "pyinstrument": "https://github.com/joerick/pyinstrument",
    "python": "https://www.python.org/",
    "pytorch": "https://pytorch.org/",
    "reverse Polish notation": "https://en.wikipedia.org/wiki/Reverse_Polish_notation",
    "scikit-learn": "https://scikit-learn.org/stable/",
    "scipy": "https://scipy.org/",
    "sklearn-onnx": "https://onnx.ai/sklearn-onnx/",
    "spox": "https://github.com/Quantco/spox",
    "sphinx-gallery": "https://github.com/sphinx-gallery/sphinx-gallery",
    "tensorflow": "https://www.tensorflow.org/",
    "tensorflow-onnx": "https://github.com/onnx/tensorflow-onnx",
    "torch": "https://pytorch.org/docs/stable/torch.html",
    "torch.onnx": "https://pytorch.org/docs/stable/onnx.html",
    #
    "C_OrtValue": (
        "http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/"
        "api/onnxruntime_python/ortvalue.html#c-class-ortvalue-or-c-ortvalue"
    ),
    "OrtValue": (
        "http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/"
        "api/onnxruntime_python/ortvalue.html#onnxruntime.OrtValue"
    ),
}
