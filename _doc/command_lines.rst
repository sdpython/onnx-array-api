=============
command lines
=============

compare
=======

The function convers an onnx file into some code.

::

    python -m compare -m1 model1.onnx -m2 model2.onnx -v 1

Output example::

    [compare_onnx_execution] got 2 inputs
    [compare_onnx_execution] execute first model
    [compare_onnx_execution] got 5 results
    [compare_onnx_execution] execute second model
    [compare_onnx_execution] got 5 results
    [compare_onnx_execution] compute edit distance
    [compare_onnx_execution] got 4 pairs
    [compare_onnx_execution] done
    = | INPUT  float32  5x6             AAAA          X    | INPUT  float32  5x6             AAAA          X   
    = | INPUT  float32  5x6             AAAA          Y    | INPUT  float32  5x6             AAAA          Y   
    = | RESULT float32  5x6             AABB Add      res  | RESULT float32  5x6             AABB Add      res 
    = | RESULT float32  5x6             AAAA Cos      Z    | RESULT float32  5x6             AAAA Cos      Z 

.. runpython::

    from onnx_array_api._command_lines_parser import get_parser_compare
    get_parser_compare().print_help()

See function :func:`onnx_array_api.reference.compare_onnx_execution`.

translate
=========

The function convers an onnx file into some code.

::

    python -m translate ...

Output example::

    not yet ready  

.. runpython::

    from onnx_array_api._command_lines_parser import get_parser_translate
    get_parser_translate().print_help()
