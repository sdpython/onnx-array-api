export ARRAY_API_TESTS_MODULE=onnx_array_api.array_api.onnx_numpy
pytest ../array-api-tests/array_api_tests/test_creation_functions.py::test_arange || exit 1
pytest ../array-api-tests/array_api_tests/test_creation_functions.py::test_ones || exit 1
pytest ../array-api-tests/array_api_tests/test_creation_functions.py::test_zeros || exit 1
