export ARRAY_API_TESTS_MODULE=onnx_array_api.array_api.onnx_numpy
pytest ../array-api-tests/array_api_tests/test_creation_functions.py::test_asarray_scalars || exit 1
pytest ../array-api-tests/array_api_tests/test_creation_functions.py --skips-file=_unittests/onnx-numpy-skips.txt -v || exit 1