from typing import Any, Optional
import pandas
import matplotlib.pyplot as plt


def plot_ort_profile(
    df: pandas.DataFrame,
    ax0: Optional[Any] = None,
    ax1: Optional[Any] = None,
    title: Optional[str] = None,
) -> Any:
    """
    Plots time spend in computation based on dataframe
    produced by function :func:`ort_profile
    <onnx_array_api.ort.ort_profile.ort_profile>`.

    :param df: dataframe
    :param ax0: first axis to draw time
    :param ax1: second axis to draw occurences
    :param title: graph title
    :return: ax0

    See :ref:`l-example-ort-profiling` for an example.
    """
    if ax0 is None:
        ax0 = plt.gca()  # pragma: no cover

    gr_dur = (
        df[["dur", "args_op_name"]].groupby("args_op_name").sum().sort_values("dur")
    )
    gr_dur.plot.barh(ax=ax0)
    if title is not None:
        ax0.set_title(title)
    if ax1 is not None:
        gr_n = (
            df[["dur", "args_op_name"]]
            .groupby("args_op_name")
            .count()
            .sort_values("dur")
        )
        gr_n = gr_n.loc[gr_dur.index, :]
        gr_n.plot.barh(ax=ax1)
        ax1.set_title("n occurences")
    return ax0
