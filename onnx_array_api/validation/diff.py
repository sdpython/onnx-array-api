import os
import difflib
import textwrap
from typing import Union
from onnx import ModelProto


def _get_diff_template():
    import jinja2

    tpl = textwrap.dedent(
        """
            <div id="{{ div_name }}"></div>
            <link rel="stylesheet" type="text/css" href="__PATH__/diff2html.min.css" />
            <script type="text/javascript" src="__PATH__/diff2html-ui-slim.min.js">
            </script>
            <script>
            const diffString = `
            --- a/{{ title }}{{ version1 }}
            +++ b/{{ title }}{{ version2 }}
            @@ -1 +1 @@
            {{ diff_content }}
            `;
            document.addEventListener('DOMContentLoaded', function () {
            var targetElement = document.getElementById('{{ div_name }}');
            var configuration = {
                drawFileList: true,
                fileListToggle: false,
                fileListStartVisible: false,
                fileContentToggle: false,
                matching: 'lines',
                outputFormat: 'line-by-line',
                synchronisedScroll: true,
                highlight: true,
                renderNothingWhenEmpty: false,
            };
            var diff2htmlUi = new Diff2HtmlUI(targetElement, diffString, configuration);
            diff2htmlUi.draw();
            diff2htmlUi.highlightCode();
            });
            </script>
            """
    )
    path = os.path.abspath(os.path.dirname(__file__))
    path = path.replace("\\", "/")
    path = f"file://{path}"
    tpl = tpl.replace("__PATH__", path)
    return jinja2.Template(tpl, autoescape=True)


def text_diff(text1: Union[ModelProto, str], text2: Union[ModelProto, str]) -> str:
    """
    Produces a string showing the differences between
    two strings.

    :param text1: first string
    :param text2: second string
    :return: differences
    """
    if not isinstance(text1, str):
        from ..plotting.text_plot import onnx_simple_text_plot

        text1 = onnx_simple_text_plot(text1, indent=False)
    if not isinstance(text2, str):
        from ..plotting.text_plot import onnx_simple_text_plot

        text2 = onnx_simple_text_plot(text2, indent=False)
    differ = difflib.Differ()
    result = list(
        differ.compare(text1.splitlines(keepends=True), text2.splitlines(keepends=True))
    )
    raw = "".join(result)
    return raw


def html_diff(
    text1: Union[ModelProto, str],
    text2: Union[ModelProto, str],
    title: str = "html_diff",
    div_name: str = "div_name",
    header: bool = True,
) -> str:
    """
    Produces a HTML files showing the differences between
    two strings.

    :param text1: first string
    :param text2: second string
    :param title: title
    :param div: html format, section name
    :param header: if True, add header and html main tags
    :return: differences
    """
    raw = text_diff(text1, text2)
    diff = _get_diff_template().render(
        title=title,
        version1=text1,
        version2=text2,
        div_name=f"div_{div_name}",
        diff_content=raw,
    )
    return f"<html><body>\n{diff}\n</body></html>\n"
