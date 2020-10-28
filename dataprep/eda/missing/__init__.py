"""
    This module implements the plot_missing(df) function
"""

from typing import Optional, Union, Dict, List, Any

import dask.dataframe as dd
import pandas as pd

from ..dtypes import DTypeDef
from ..progress_bar import ProgressBar
from .compute import compute_missing
from .render import render_missing
from ..container import Container
from ..basic.configs import Config

__all__ = ["render_missing", "compute_missing", "plot_missing"]


def plot_missing(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    bins: int = 20,
    ndist_sample: int = 100,
    dtype: Optional[DTypeDef] = None,
    progress: bool = True,
    config: Union[Dict[str, Any], str] = "auto",
    display: Union[List[str], str] = "auto",
) -> Container:
    """
    This function is designed to deal with missing values
    There are three functions: plot_missing(df), plot_missing(df, x)
    plot_missing(df, x, y)

    Parameters
    ----------
    df
        the pandas data_frame for which plots are calculated for each column.
    x
        a valid column name of the data frame.
    y
        a valid column name of the data frame.
    bins
        The number of rows in the figure.
    ndist_sample
        The number of sample points.
    wdtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous().
    progress
        Enable the progress bar.
    config
        The config dict user passed in. E.g. config =  {"hist.bins": 20}
        Without user's specifications, the default is "auto"
    display
        The list that contains the names of plots user wants to display,
        E.g. display =  ["bar", "hist"]
        Without user's specifications, the default is "auto"
    Examples
    ----------
    >>> from dataprep.eda.missing.computation import plot_missing
    >>> import pandas as pd
    >>> df = pd.read_csv("suicide-rate.csv")
    >>> plot_missing(df, "HDI_for_year")
    >>> plot_missing(df, "HDI_for_year", "population")
    """
    cfg = Config.from_dict(display, config)
    with ProgressBar(minimum=1, disable=not progress):
        itmdt = compute_missing(df, x, y, dtype=dtype, bins=bins, ndist_sample=ndist_sample)
    to_render = render_missing(itmdt)

    return Container(to_render, itmdt.visual_type, cfg=cfg)
