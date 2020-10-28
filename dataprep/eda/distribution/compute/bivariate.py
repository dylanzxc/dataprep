"""Computations for plot(df, x, y)."""
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ....errors import UnreachableError
from ...intermediate import Intermediate
from ...dtypes import (
    Continuous,
    DateTime,
    DTypeDef,
    Nominal,
    detect_dtype,
    is_dtype,
)
from .common import (
    DTMAP,
    _get_timeunit,
    _calc_line_dt,
    _calc_groups,
    _calc_box_otlrs,
    _calc_box_stats,
)
from ...basic.configs import Config

# pylint: disable=protected-access


def compute_bivariate(
    df: dd.DataFrame, cfg: Config, x: str, y: str, dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """Compute functions for plot(df, x, y).

    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    x
        A valid column name from the dataframe
    y
        A valid column name from the dataframe
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    cfg:
        Config instance created using config and display that user passed in.
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements

    xtype = detect_dtype(df[x], dtype)
    ytype = detect_dtype(df[y], dtype)
    if (
        is_dtype(xtype, Nominal())
        and is_dtype(ytype, Continuous())
        or is_dtype(xtype, Continuous())
        and is_dtype(ytype, Nominal())
    ):
        x, y = (x, y) if is_dtype(xtype, Nominal()) else (y, x)
        df = df[[x, y]]
        first_rows = df.head()
        try:
            first_rows[x].apply(hash)
        except TypeError:
            df[x] = df[x].astype(str)
        # later the input arguments will only include cfg after we finish the how-to-guide.
        (comps,) = dask.compute(
            nom_cont_comps(
                df.dropna(),
                cfg.line.bins,
                cfg.boxplot.ngroups,
                cfg.line.ngroups,
                cfg.boxplot.sort_descending,
                cfg.line.sort_descending,
                cfg,
            )
        )
        return Intermediate(
            x=x,
            y=y,
            data=comps,
            ngroups_box=cfg.boxplot.ngroups,
            ngroups_line=cfg.line.ngroups,
            visual_type="cat_and_num_cols",
        )
    elif (
        is_dtype(xtype, DateTime())
        and is_dtype(ytype, Continuous())
        or is_dtype(xtype, Continuous())
        and is_dtype(ytype, DateTime())
    ):
        x, y = (x, y) if is_dtype(xtype, DateTime()) else (y, x)
        df = df[[x, y]].dropna()
        dtnum: List[Any] = []
        # line chart
        if cfg.line._enable:
            dtnum.append(dask.delayed(_calc_line_dt)(df, cfg.line.unit, cfg.line.agg))
        # box plot
        if cfg.boxplot._enable:
            dtnum.append(dask.delayed(calc_box_dt)(df, cfg.boxplot.unit))
        dtnum = dask.compute(*dtnum)

        if len(dtnum) == 2:
            linedata = dtnum[0]
            boxdata = dtnum[1]
        elif cfg.line._enable:
            linedata = dtnum[0]
            boxdata = []
        else:
            boxdata = dtnum[0]
            linedata = []

        return Intermediate(
            x=x, y=y, linedata=linedata, boxdata=boxdata, visual_type="dt_and_num_cols",
        )
    elif (
        is_dtype(xtype, DateTime())
        and is_dtype(ytype, Nominal())
        or is_dtype(xtype, Nominal())
        and is_dtype(ytype, DateTime())
    ):
        x, y = (x, y) if is_dtype(xtype, DateTime()) else (y, x)
        df = df[[x, y]].dropna()
        df[y] = df[y].apply(str, meta=(y, str))
        dtcat: List[Any] = []
        if cfg.line._enable:
            # line chart
            dtcat.append(
                dask.delayed(_calc_line_dt)(
                    df, cfg.line.unit, ngroups=cfg.line.ngroups, largest=cfg.line.sort_descending
                )
            )
        if cfg.stackedbar._enable:
            # stacked bar chart
            dtcat.append(
                dask.delayed(calc_stacked_dt)(
                    df, cfg.stackedbar.unit, cfg.stackedbar.ngroups, cfg.stackedbar.sort_descending
                )
            )
        dtcat = dask.compute(*dtcat)

        if len(dtcat) == 2:
            linedata = dtcat[0]
            stackdata = dtcat[1]
        elif cfg.line._enable:
            linedata = dtcat[0]
            stackdata = []
        else:
            stackdata = dtcat[0]
            linedata = []

        return Intermediate(
            x=x, y=y, linedata=linedata, stackdata=stackdata, visual_type="dt_and_cat_cols",
        )
    elif is_dtype(xtype, Nominal()) and is_dtype(ytype, Nominal()):
        df = df[[x, y]]
        first_rows = df.head()
        try:
            first_rows[x].apply(hash)
        except TypeError:
            df[x] = df[x].astype(str)
        try:
            first_rows[y].apply(hash)
        except TypeError:
            df[y] = df[y].astype(str)

        (comps,) = dask.compute(df.dropna().groupby([x, y]).size())
        return Intermediate(
            x=x,
            y=y,
            data=comps,
            ngroups_nested=cfg.nestedbar.ngroups,
            nsubgroups_nested=cfg.nestedbar.nsubgroups,
            ngroups_stacked=cfg.stackedbar.ngroups,
            nsubgroups_stacked=cfg.stackedbar.nsubgroups,
            ngroups_heat=cfg.heatmap.ngroups_x,
            nsubgroups_heat=cfg.heatmap.ngroups_y,
            visual_type="two_cat_cols",
        )
    elif is_dtype(xtype, Continuous()) and is_dtype(ytype, Continuous()):
        # one partition required for apply(pd.cut) in calc_box_num
        df = df[[x, y]].dropna().repartition(npartitions=1)

        data: Dict[str, Any] = {}
        if cfg.scatter._enable:
            # scatter plot data
            data["scat"] = df.map_partitions(lambda x: x.sample(min(100, x.shape[0])), meta=df)
        if cfg.hexbin._enable:
            # hexbin plot data
            data["hex"] = df
        if cfg.boxplot._enable:
            # box plot
            data["box"] = calc_box_num(df, cfg.boxplot.bins)

        (data,) = dask.compute(data)

        return Intermediate(
            x=x, y=y, data=data, spl_sz=cfg.scatter.sample_size, visual_type="two_num_cols",
        )
    else:
        raise UnreachableError


def nom_cont_comps(
    df: dd.DataFrame,
    bins: int,
    ngroups_box: int,
    ngroups_hist: int,
    largest_box: bool,
    largest_hist: bool,
    cfg: Config,
) -> Dict[str, Any]:
    """
    Computations for a nominal and continuous column

    Parameters
    ----------
    df
        Dask dataframe with one categorical and one numerical column
    bins
        Number of bins to use in the histogram, later be used to form a line chart,
        so here we use the line config
    ngroups_box
        Number of groups to show from the categorical column for the box plot
    ngroups_hist
        Number of groups to show from the categorical column for the histogram,
        later be used to form a line chart, so here we use the line config
    largest_box
        Select the largest or smallest groups for the box plot
    largest_hist
        Select the largest or smallest groups for the histogram, later be used to form a line chart,
         so here we use the line config
    """
    # pylint: disable = too-many-arguments,too-many-locals
    data: Dict[str, Any] = {}

    x, y = df.columns[0], df.columns[1]

    # filter the dataframe to consist of ngroup groups
    # https://stackoverflow.com/questions/46927174/filtering-grouped-df-in-dask
    cnts = df[x].value_counts(sort=False)
    data["ttl_grps"] = cnts.shape[0]
    thresh_box = (
        cnts.nlargest(ngroups_box).min() if largest_box else cnts.nsmallest(ngroups_box).max()
    )
    df_box = df[df[x].map(cnts) >= thresh_box] if largest_box else df[df[x].map(cnts) <= thresh_box]

    # group the data to compute a box plot and histogram for each group
    grps_box = df_box.groupby(x)[y]
    if cfg.boxplot._enable:
        data["box"] = grps_box.apply(box_comps, meta="object")
    if cfg.line._enable:
        if largest_box == largest_hist and ngroups_box == ngroups_hist:
            minv, maxv = df_box[y].min(), df_box[y].max()
            # TODO when are minv and maxv computed? This may not be optimal if
            # minv and maxv are computed ngroups times for each histogram
            data["hist"] = grps_box.apply(hist, bins, minv, maxv, meta="object")
        else:
            thresh_hist = (
                cnts.nlargest(ngroups_hist).min()
                if largest_hist
                else cnts.nsmallest(ngroups_hist).max()
            )
            df_hist = (
                df[df[x].map(cnts) >= thresh_hist]
                if largest_hist
                else df[df[x].map(cnts) <= thresh_hist]
            )
            grps_hist = df_hist.groupby(x)[y]
            minv, maxv = df_hist[y].min(), df_hist[y].max()
            data["hist"] = grps_hist.apply(hist, bins, minv, maxv, meta="object")

    return data


def calc_box_num(df: dd.DataFrame, bins: int) -> dd.Series:
    """
    Box plot for a binned numerical variable

    Parameters
    ----------
    df
        dask dataframe
    bins
        number of bins to compute a box plot
    """
    x, y = df.columns[0], df.columns[1]
    # group the data into intervals
    # https://stackoverflow.com/questions/42442043/how-to-use-pandas-cut-or-equivalent-in-dask-efficiently
    df["grp"] = df[x].map_partitions(pd.cut, bins=bins, include_lowest=True)
    # TODO is this calculating the box plot stats for each group in parallel?
    # https://examples.dask.org/dataframes/02-groupby.html#Groupby-Apply
    # https://github.com/dask/dask/issues/4239
    # https://github.com/dask/dask/issues/5124
    srs = df.groupby("grp")[y].apply(box_comps, meta="object")

    return srs


def box_comps(srs: pd.Series) -> Dict[str, Union[float, np.array]]:
    """
    Box plot computations

    Parameters
    ----------
    srs
        pandas series
    """
    data: Dict[str, Any] = {}

    # quartiles
    data.update(zip(("q1", "q2", "q3"), srs.quantile([0.25, 0.5, 0.75])))
    iqr = data["q3"] - data["q1"]
    # inliers
    srs_iqr = srs[srs.between(data["q1"] - 1.5 * iqr, data["q3"] + 1.5 * iqr)]
    data["lw"], data["uw"] = srs_iqr.min(), srs_iqr.max()
    # outliers
    otlrs = srs[~srs.between(data["q1"] - 1.5 * iqr, data["q3"] + 1.5 * iqr)]
    # randomly sample at most 100 outliers
    data["otlrs"] = otlrs.sample(min(100, otlrs.shape[0])).values

    return data


def hist(srs: pd.Series, bins: int, minv: float, maxv: float) -> Any:
    """
    Compute a histogram on a given series

    Parameters
    ----------
    srs
        pandas Series of values for the histogram
    bins
        number of bins
    minv
        lowest bin endpoint
    maxv
        highest bin endpoint
    """

    return np.histogram(srs, bins=bins, range=[minv, maxv])


def calc_box_dt(df: dd.DataFrame, unit: str) -> Tuple[pd.DataFrame, List[str], List[float], str]:
    """
    Calculate a box plot with date on the x axis.
    Parameters
    ----------
    df
        A dataframe with one datetime and one numerical column
    unit
        The unit of time over which to group the values
    """

    x, y = df.columns[0], df.columns[1]  # time column
    unit = _get_timeunit(df[x].min(), df[x].max(), 10) if unit == "auto" else unit
    if unit not in DTMAP.keys():
        raise ValueError
    grps = df.groupby(pd.Grouper(key=x, freq=DTMAP[unit][0]))  # time groups
    # box plot for the values in each time group
    df = pd.concat([_calc_box_stats(g[1][y], g[0], True) for g in grps], axis=1,)
    df = df.append(pd.Series({c: i + 1 for i, c in enumerate(df.columns)}, name="x",)).T
    # If grouping by week, make the label for the week the beginning Sunday
    df.index = df.index - pd.to_timedelta(6, unit="d") if unit == "week" else df.index
    df.index.name = "grp"
    df = df.reset_index()
    df["grp"] = df["grp"].dt.to_period("S").dt.strftime(DTMAP[unit][2])
    df["x0"], df["x1"] = df["x"] - 0.8, df["x"] - 0.2  # width of whiskers for plotting
    outx, outy = _calc_box_otlrs(df)

    return df, outx, outy, DTMAP[unit][3]


def calc_stacked_dt(
    df: dd.DataFrame, unit: str, ngroups: int, largest: bool,
) -> Tuple[pd.DataFrame, Dict[str, int], str]:
    """
    Calculate a stacked bar chart with date on the x axis
    Parameters
    ----------
    df
        A dataframe with one datetime and one categorical column
    unit
        The unit of time over which to group the values
    ngroups
        Number of groups for the categorical column
    largest
        Use the largest or smallest groups in the categorical column
    """
    # pylint: disable=too-many-locals

    x, y = df.columns[0], df.columns[1]  # time column
    unit = _get_timeunit(df[x].min(), df[x].max(), 10) if unit == "auto" else unit
    if unit not in DTMAP.keys():
        raise ValueError

    # get the largest groups
    df_grps, grp_cnt_stats, _ = _calc_groups(df, y, ngroups, largest)
    grouper = (pd.Grouper(key=x, freq=DTMAP[unit][0]),)  # time grouper
    # pivot table of counts with date groups as index and categorical values as column names
    dfr = pd.pivot_table(df_grps, index=grouper, columns=y, aggfunc=len, fill_value=0,).rename_axis(
        None
    )

    # if more than ngroups categorical values, aggregate the smallest groups into "Others"
    if grp_cnt_stats[f"{y}_ttl"] > grp_cnt_stats[f"{y}_shw"]:
        grp_cnts = df.groupby(pd.Grouper(key=x, freq=DTMAP[unit][0])).size()
        dfr["Others"] = grp_cnts - dfr.sum(axis=1)

    dfr.index = (  # If grouping by week, make the label for the week the beginning Sunday
        dfr.index - pd.to_timedelta(6, unit="d") if unit == "week" else dfr.index
    )
    dfr.index = dfr.index.to_period("S").strftime(DTMAP[unit][2])  # format labels

    return dfr, grp_cnt_stats, DTMAP[unit][3]
