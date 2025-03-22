import sys
import onnx
from typing import Any, List, Optional
from argparse import ArgumentParser
from textwrap import dedent


def get_main_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="onnx-array-api",
        description="onnx-array-api main command line.",
        epilog="Type 'python -m onnx_array_api <cmd> --help' "
        "to get help for a specific command.",
    )
    parser.add_argument(
        "cmd",
        choices=["translate", "compare", "replace"],
        help=dedent(
            """
        Selects a command.

        'translate' exports an onnx graph into a piece of code replicating it,
        'compare' compares the execution of two onnx models,
        'replace' replaces constant and initliazers by ConstantOfShape
                  to make the model lighter
        """
        ),
    )
    return parser


def get_parser_translate() -> ArgumentParser:
    parser = ArgumentParser(
        prog="translate",
        description=dedent(
            """
        Translates an onnx model into a piece of code to replicate it.
        The result is printed on the standard output.
        """
        ),
        epilog="This is mostly used to write unit tests without adding "
        "an onnx file to the repository.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="onnx model to translate",
    )
    parser.add_argument(
        "-a",
        "--api",
        choices=["onnx", "light", "onnx-short", "builder"],
        default="onnx",
        help="API to choose, API from onnx package or light API.",
    )
    return parser


def _cmd_translate(argv: List[Any]):
    from .translate_api import translate

    parser = get_parser_translate()
    args = parser.parse_args(argv[1:])
    onx = onnx.load(args.model)
    code = translate(onx, api=args.api)
    print(code)


def get_parser_compare() -> ArgumentParser:
    parser = ArgumentParser(
        prog="compare",
        description=dedent(
            """
        Compares the execution of two onnx models.
        """
        ),
        epilog="This is used when two models are different but "
        "should produce the same results.",
    )
    parser.add_argument(
        "-m1",
        "--model1",
        type=str,
        required=True,
        help="first onnx model",
    )
    parser.add_argument(
        "-m2",
        "--model2",
        type=str,
        required=True,
        help="second onnx model",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["execute", "nodes"],
        default="execute",
        help="compare the execution ('execute') or the nodes only ('nodes')",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        help="verbosity",
    )
    parser.add_argument(
        "-c",
        "--column-size",
        default=60,
        help="column size when displaying the results",
    )
    parser.add_argument(
        "-d",
        "--discrepancies",
        default=0,
        help="show precise discrepancies when mode is execution",
    )
    return parser


def _cmd_compare(argv: List[Any]):
    from .reference import compare_onnx_execution

    parser = get_parser_compare()
    args = parser.parse_args(argv[1:])
    if args.verbose in ("1", 1, "True", True):
        print(f"[compare] first model {args.model1!r}")
        print(f"[compare] second model {args.model2!r}")
    onx1 = onnx.load(args.model1)
    onx2 = onnx.load(args.model2)
    if args.verbose in ("1", 1, "True", True):
        print(f"[compare] first model has {len(onx1.graph.node)} nodes")
        print(f"[compare] second model has {len(onx2.graph.node)} nodes")
    res1, res2, align, dc = compare_onnx_execution(
        onx1,
        onx2,
        verbose=args.verbose,
        mode=args.mode,
        keep_tensor=args.discrepancies in (1, "1", "True", True),
    )
    text = dc.to_str(res1, res2, align, column_size=int(args.column_size))
    print(text)


def get_parser_replace() -> ArgumentParser:
    parser = ArgumentParser(
        prog="translate",
        description=dedent(
            """
        Replaces constants and initializes by ConstOfShape or any other nodes
        to make the model smaller.
        """
        ),
        epilog="This is mostly used to write unit tests without adding "
        "a big file to the repository.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="onnx model to translate",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        required=True,
        help="output file",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=128,
        help="Threshold above which every constant is replaced",
    )
    parser.add_argument(
        "--type",
        default="ConstontOfShape",
        help="Inserts this operator type",
    )
    parser.add_argument(
        "--domain",
        default="",
        help="Inserts this domain",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        help="verbosity",
    )
    return parser


def _cmd_replace(argv: List[Any]):
    from .tools.replace_constants import replace_initializer_by_constant_of_shape

    parser = get_parser_replace()
    args = parser.parse_args(argv[1:])
    if args.verbose in ("1", 1, "True", True):
        print(f"[compare] load model {args.model!r}")
    onx = onnx.load(args.model)
    new_onx = replace_initializer_by_constant_of_shape(
        onx, threshold=args.threshold, op_type=args.type, domain=args.domain
    )
    if args.verbose in ("1", 1, "True", True):
        print(f"[compare] save model {args.out!r}")
    onnx.save(new_onx, args.out)


def main(argv: Optional[List[Any]] = None):
    fcts = dict(translate=_cmd_translate, compare=_cmd_compare, replace=_cmd_replace)

    if argv is None:
        argv = sys.argv[1:]
    if (len(argv) <= 1 and argv[0] not in fcts) or argv[-1] in ("--help", "-h"):
        if len(argv) < 2:
            parser = get_main_parser()
            parser.parse_args(argv)
        else:
            parsers = dict(
                translate=get_parser_translate,
                compare=get_parser_compare,
                replace=get_parser_replace,
            )
            cmd = argv[0]
            if cmd not in parsers:
                raise ValueError(
                    f"Unknown command {cmd!r}, it should be in {list(sorted(parsers))}."
                )
            parser = parsers[cmd]()
            parser.parse_args(argv[1:])
        raise RuntimeError("The programme should have exited before.")

    cmd = argv[0]
    if cmd in fcts:
        fcts[cmd](argv)
    else:
        raise ValueError(
            f"Unknown command {cmd!r}, use --help to get the list of known command."
        )
