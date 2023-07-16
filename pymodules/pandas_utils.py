"""Pandas utilities."""
from __future__ import annotations

import glob
import logging
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import io as sio

from pymodules.file_utils import glob_re, logger

logger = logging.getLogger(__name__)


def format_series_name(series_name: str | Tuple, sep: str = " ") -> str:
    """Returns series name as a string."""

    if isinstance(series_name, tuple):
        return sep.join((str(bit) for bit in series_name))
    return str(series_name)


def extract_calendar_features(df: pd.DataFrame, level: str | None = None) -> pd.DataFrame:
    """Generates calendar features from a DatetimeIndex.

    Args:
        df:
            A DataFrame with a DatetimeIndex level.
        level:
            If not provided, index is parsed to find a DatetimeIndex level.

    Returns:
        A DataFrame with a MultiIndex.
    """

    # Isolate the datetime index
    if level is None:
        is_index_level_datetime = [
            isinstance(
                df.index.get_level_values(level),
                pd.core.indexes.datetimes.DatetimeIndex,
            )
            for level in df.index.names
        ]
        if sum(is_index_level_datetime) == 0:
            raise TypeError("A DatetimeIndex level is required")

        if sum(is_index_level_datetime) > 1:
            logger.warning(
                f"More than one DatetimeIndex level is available, the first one: "
                f"{df.index.names[is_index_level_datetime.index(True)]} will be used"
            )

        # Isolate the level
        datetime_index = df.index.get_level_values(
            df.index.names[np.where(is_index_level_datetime)[0][0]]
        )
    else:
        datetime_index = df.index.get_level_values(level)

    # Weekday dictionary
    weekday_dict = dict(
        zip(
            range(7),
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )
    )
    # Calendar features
    hour = pd.Index(datetime_index.hour, name="hour")
    minute = pd.Index(datetime_index.minute, name="minute")
    day = pd.Index(datetime_index.day, name="day")
    dow = pd.Index(datetime_index.weekday, name="dow")
    weekday = pd.Index(dow.map(lambda x: weekday_dict[x]), name="weekday")
    month = pd.Index(datetime_index.month, name="month")
    date = pd.Index(datetime_index.date, name="date")

    # Return a multi-indexed DataFrame
    df = df.copy()
    # Join the new levels onto the originals
    df.index = pd.MultiIndex.from_arrays(
        (
            *[
                calendar_feat
                for calendar_feat in [
                    minute,
                    hour,
                    day,
                    dow,
                    weekday,
                    month,
                    date,
                ]
                if calendar_feat.name not in df.index.names
            ],
            *[df.index.get_level_values(level) for level in df.index.names],
        )
    )

    return df


def flatten_index(df: pd.DataFrame, axis: str | int = 1, sep: str = " ") -> pd.DataFrame:
    """Flatten a hierarchical index in a human-friendly way."""
    flatten_df = df.copy()

    if axis not in [0, 1, "both"]:
        raise ValueError(f"Invalid axis value: {axis} is not one of [0, 1, 'both']")

    if (axis == 1) | (axis == "both"):
        index_names = [
            name if name else "level" + str(i) for i, name in enumerate(df.columns.names)
        ]
        flatten_df.columns = pd.Index(
            [sep.join(map(lambda x: str(x), col)).strip() for col in flatten_df.columns],
            name="_".join(index_names),
        )

    if (axis == 0) | (axis == "both"):
        index_names = [name if name else "level" + str(i) for i, name in enumerate(df.index.names)]
        flatten_df.index = pd.Index(
            [sep.join(map(lambda x: str(x), col)).strip() for col in flatten_df.index],
            name="_".join(index_names),
        )

    return flatten_df


def get_pd_name(panda_object: pd.Series | pd.DataFrame) -> List[str]:
    """Returns column or series name of the panda object."""
    try:
        names = list(panda_object.columns)
    except AttributeError:
        names = list((panda_object.name,))

    return names


def join_csv(csv_path: str, pattern: str | None = None) -> pd.DataFrame:
    """Joins multiple csv files with same columns into a panda DataFrame."""
    # List the csv
    csv_list = glob.glob(os.path.join(csv_path, "*.csv"))
    # Filter with the pattern if applicable
    if pattern:
        csv_list = glob_re(pattern, csv_list)

    return pd.concat([pd.read_csv(f) for f in csv_list])


def alternate_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Inserts a blank row after between every row of a DataFrame."""
    nans = np.empty_like(df.values, dtype=float)
    nans.fill(np.nan)
    data = np.hstack([nans, df.values]).reshape(-1, df.shape[1])
    return pd.DataFrame(data, columns=df.columns)


def coalesce(
    df: pd.DataFrame, column_names: List | None = None, name: str | None = None
) -> pd.DataFrame:
    """Combines columns from left to right with filling to coalesce columns
    SQL-like.
    """
    if column_names is None:
        column_names = list(df.columns)
    i = iter(column_names)
    column_name = next(i)
    answer = df[column_name]
    for column_name in i:
        answer = answer.fillna(df[column_name])
    # Drop the columns we coalesced
    df = df.drop(columns=column_names)
    # Rename the coalesced column
    if name is not None:
        if isinstance(df.columns, pd.MultiIndex) and not isinstance(name, tuple):
            raise ValueError(f"Invalid column name: '{name}' for a MultiIndex")
        answer = answer.rename(name)

    # Return the DataFrame with the coalesced column
    return df.join(answer)


def regexp_columns(
    pattern: str,
    df: pd.DataFrame,
    full_label: bool = False,
    case_sensitive: bool = False,
) -> List:
    """Returns a list of panda columns matching regexp patterns.

    Args:
        full_label:
            If True, match only if the pattern entirely match the column label.
        case_sensitive:
            If True, match is case-sensitive.

    """
    # handle hierarchical index
    if df.columns.nlevels == 1:
        columns_list = df.columns
    else:
        columns_list = ["_".join(col).strip() for col in df.columns]

    if full_label:
        pattern = f"^{pattern}$"

    if not case_sensitive:
        pattern = f"(?i){pattern}"

    # Return the columns that match the pattern
    selected_columns = df.columns[
        [match is not None for match in map(lambda x: re.search(pattern, x), columns_list)]
    ]

    return list(selected_columns)


def unique_non_null(series: pd.Series) -> pd.Series:
    """Returns unique non-null values."""
    return series.dropna().unique()


def pandas_to_matlab(df: pd.DataFrame, filename: str) -> None:
    """Saves a Pandas DataFrame to a matlab-compatible dictionary of numpy arraus."""
    # Reset index to preserve it
    df = df.reset_index()

    pandas_dict = {col_name: df[col_name].values for col_name in df.columns}

    # Finally save to a matfile
    try:
        sio.savemat(filename, pandas_dict)
    except IOError as exc:
        logger.error(f"I/O error: {exc}")
    else:
        logger.info(
            f"Saved DataFrame with {list(pandas_dict.keys())} columns to {filename} matfile"
        )
