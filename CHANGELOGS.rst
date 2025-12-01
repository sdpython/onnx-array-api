Change Logs
===========

0.3.4
+++++

0.3.3
+++++

* :pr:`104`: add code rendering when conveting a model into code
* :pr:`103`: fix import issue with the latest onnx version

0.3.2
+++++

* :pr:`101`: fix as_tensor in onnx_text_plot_tree

0.3.1
+++++

* :pr:`100`: updates requirements, add 3.12
* :pr:`96`: supports local functions in translator
* :pr:`95`: improves translation to GraphBuilder

0.3.0
+++++

* :pr:`93`: fixes evaluator type in ``compare_onnx_execution``
* :pr:`92`: avoids recursion errors in profiling
* :pr:`87`: adds command line to replace contant by ConstantOfShape
* :pr:`79`: first draft to export to GraphBuilder
* :pr:`77`: supports ConcatOfShape and Slice with the light API

0.2.0
+++++

* :pr:`76`, :pr:`79`: add a mode to compare models without execution
* :pr:`75`: add QuickGelu to ExtendedReferenceEvaluator
* :pr:`71`: adds tools to compare two onnx graphs
* :pr:`61`: adds function to plot onnx model as graphs
* :pr:`60`: supports translation of local functions
* :pr:`59`: add methods to update nodes in GraphAPI 

0.1.3
+++++

* :pr:`57`: implements GraphBuilder
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
