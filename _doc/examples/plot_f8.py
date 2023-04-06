"""
.. _l-example-float8:

About float 8
=============

Float 8 types were recently introduced to speed up the
training of deep learning models.

Possible values
+++++++++++++++

First E4M3FN.
"""

import pprint
from onnx_array_api.validation.f8 import CastFloat8

pprint.pprint(CastFloat8.values_e4m3fn)


############################################
# Then E5M2.

pprint.pprint(CastFloat8.values_e5m2)
