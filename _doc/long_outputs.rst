==========================
Long outputs uneasy to see
==========================

onnx
====

.. _l-long-output-compare_onnx_execution:

onnx_array_api.reference.compare_onnx_execution
+++++++++++++++++++++++++++++++++++++++++++++++

From example :ref:`l-onnx-diff-example` for function
:func:`onnx_array_api.reference.compare_onnx_execution`.

::

    1 = | INITIA float64  1               HAAA            Ad_Addcst    | INITIA float64  1               HAAA            Ad_Addcst   
    2 = | INITIA float64  4x4             ADZF            Ge_Gemmcst   | INITIA float64  4x4             ADZF            Ge_Gemmcst  
    3 = | INITIA float64  4               USEA            Ge_Gemmcst1  | INITIA float64  4               USEA            Ge_Gemmcst1 
    4 = | INITIA float64  1               AAAA            Mu_Mulcst    | INITIA float64  1               AAAA            Mu_Mulcst   
    5 = | INITIA float64  1               DAAA            Ad_Addcst1   | INITIA float64  1               DAAA            Ad_Addcst1  
    6 = | INITIA float64  1               AAAA            Ad_Addcst2   | INITIA float64  1               AAAA            Ad_Addcst2  
    7 = | INPUT  float64  1x4             AAAA            X            | INPUT  float64  1x4             AAAA            X           
    8 = | RESULT float64  1x4             UTFC Gemm       Ge_Y0        | RESULT float64  1x4             UTFC Gemm       Ge_Y0       
    9 + |                                                              | RESULT float64  1x4             TIEG Mul        Mu_C01       
    10 ~ | RESULT float64  1x1             NAAA ReduceSumS Re_reduced0  | RESULT float64  1x1             NAAA ReduceSum  Re_reduced0 
    11 = | RESULT float64  1x1             NAAA Concat     Co_concat_re | RESULT float64  1x1             NAAA Concat     Co_concat_re
    12 = | RESULT float64  1x1             UAAA Add        Ad_C02       | RESULT float64  1x1             UAAA Add        Ad_C02      
    13 = | RESULT float64  1x1             DAAA Mul        Mu_C0        | RESULT float64  1x1             DAAA Mul        Mu_C0       
    14 = | RESULT float64  1x1             GAAA Add        Ad_C01       | RESULT float64  1x1             GAAA Add        Ad_C01      
    15 = | RESULT float64  1x1             GAAA Add        Ad_C0        | RESULT float64  1x1             GAAA Add        Ad_C0       
    16 = | RESULT int64    1x1             AAAA ArgMax     label        | RESULT int64    1x1             AAAA ArgMax     label       
    17 + |                                                              | RESULT float64  1x1             GAAA ReduceMax  Re_reduced03 
    18 + |                                                              | RESULT float64  1x1             AAAA Sub        Su_C01       
    19 + |                                                              | RESULT float64  1x1             BAAA Exp        Ex_output0   
    20 + |                                                              | RESULT float64  1x1             BAAA ReduceSum  Re_reduced02 
    21 + |                                                              | RESULT float64  1x1             AAAA Log        Lo_output0   
    22 ~ | RESULT float64  1x1             GAAA ReduceLogS score_sample | RESULT float64  1x1             GAAA Add        score_sample
    23 = | RESULT float64  1x1             AAAA Sub        Su_C0        | RESULT float64  1x1             AAAA Sub        Su_C0       
    24 = | RESULT float64  1x1             BAAA Exp        probabilitie | RESULT float64  1x1             BAAA Exp        probabilitie
    25 = | OUTPUT int64    1x1             AAAA            label        | OUTPUT int64    1x1             AAAA            label       
    26 = | OUTPUT float64  1x1             BAAA            probabilitie | OUTPUT float64  1x1             BAAA            probabilitie
    27 = | OUTPUT float64  1x1             GAAA            score_sample | OUTPUT float64  1x1             GAAA            score_sample    
