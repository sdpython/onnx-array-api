@echo off
set ARRAY_API_TESTS_MODULE=onnx_array_api.array_api.onnx_numpy
python -m pytest ../../array-api-tests/array_api_tests/test_creation_functions.py::test_arange || exit 1
python -m pytest ../../array-api-tests/array_api_tests/test_creation_functions.py --hypothesis-explain --skips-file=_unittests/onnx-numpy-skips.txt || exit 1
