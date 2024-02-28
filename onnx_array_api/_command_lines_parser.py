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
        choices=["translate", "compare"],
        help=dedent(
            """
        Selects a command.
        
        'translate' exports an onnx graph into a piece of code replicating it,
        'compare' compares the execution of two onnx models
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
        choices=["onnx", "light"],
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
        epilog="This is used when two models are different but should produce the same results.",
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
    onx1 = onnx.load(args.model1)
    onx2 = onnx.load(args.model2)
    res1, res2, align, dc = compare_onnx_execution(
        onx1,
        onx2,
        verbose=args.verbose,
        mode=args.mode,
        keep_tensor=args.discrepancies in (1, "1", "True", True),
    )
    text = dc.to_str(res1, res2, align, column_size=int(args.column_size))
    print(text)


def main(argv: Optional[List[Any]] = None):
    fcts = dict(translate=_cmd_translate, compare=_cmd_compare)

    if argv is None:
        argv = sys.argv[1:]
    if (len(argv) <= 1 and argv[0] not in fcts) or argv[-1] in ("--help", "-h"):
        if len(argv) < 2:
            parser = get_main_parser()
            parser.parse_args(argv)
        else:
            parsers = dict(translate=get_parser_translate, compare=get_parser_compare)
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
