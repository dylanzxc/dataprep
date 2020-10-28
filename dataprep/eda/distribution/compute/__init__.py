"""Computations for plot(df, ...)."""

from typing import Optional, Tuple, Union, cast

import dask.dataframe as dd
import pandas as pd

from ...dtypes import DTypeDef, string_dtype_to_object
from ...intermediate import Intermediate
from ...utils import to_dask
from .bivariate import compute_bivariate
from .overview import compute_overview
from .trivariate import compute_trivariate
from .univariate import compute_univariate
from ...basic.configs import Config

__all__ = ["compute"]


def compute(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    dtype: Optional[DTypeDef] = None,
    cfg: Config = "auto",
) -> Intermediate:
    """All in one compute function.

    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    x: Optional[str], default None
        A valid column name from the dataframe
    y: Optional[str], default None
        A valid column name from the dataframe
    z: Optional[str], default None
        A valid column name from the dataframe
    value_range: Optional[Tuple[float, float]], default None
        The lower and upper bounds on the range of a numerical column.
        Applies when column x is specified and column y is unspecified.
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    cfg: Config instance created using config and display that user passed in.
    """  # pylint: disable=too-many-locals
    df = to_dask(df)
    df.columns = df.columns.astype(str)
    df = string_dtype_to_object(df)

    if not any((x, y, z)):
        return compute_overview(df, cfg, dtype)

    if sum(v is None for v in (x, y, z)) == 2:
        col: str = cast(str, x or y or z)
        return compute_univariate(df, col, cfg, value_range, dtype,)

    if sum(v is None for v in (x, y, z)) == 1:
        x, y = (v for v in (x, y, z) if v is not None)
        return compute_bivariate(df, cfg, x, y, dtype,)

    if x is not None and y is not None and z is not None:
        return compute_trivariate(cfg, df, x, y, z, dtype)

    raise ValueError("not possible")
