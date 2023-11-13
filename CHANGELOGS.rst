Change Logs
===========

0.1.3
+++++

* :pr:`49`: adds command line to export a model into code
* :pr:`48`: support for subgraph in light API
* :pr:`47`: extends export onnx to code to support inner API
* :pr:`46`: adds an export to convert an onnx graph into light API code
* :pr:`45`: fixes light API for operators with two outputs

0.1.2
+++++

* :pr:`42`: first sketch for a very simple API to create onnx graph in one or two lines
* :pr:`27`: add function from_array_extended to convert
  an array to a TensorProto, including bfloat16 and float 8 types
* :pr:`24`: add ExtendedReferenceEvaluator to support scenario
  for the Array API onnx does not support
* :pr:`22`: support OrtValue in function *ort_profile*
* :pr:`17`: implements ArrayAPI
* :pr:`3`: fixes Array API with onnxruntime and scikit-learn
