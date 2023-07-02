export ARRAY_API_TESTS_MODULE=onnx_array_api.array_api.onnx_numpy
pytest -v -rxXfE ../array-api-tests/array_api_tests/test_creation_functions.py::test_eye || exit 1
# pytest ../array-api-tests/array_api_tests/test_creation_functions.py --help
pytest -v -rxXfE ../array-api-tests/array_api_tests/test_creation_functions.py --hypothesis-explain --skips-file=_unittests/onnx-numpy-skips.txt || exit 1
