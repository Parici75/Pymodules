import logging

import numpy as np
import pandas as pd
import pytest

from pymodules.pandas_utils import (
    flatten_index,
    format_series_name,
    pandas_to_matlab,
    extract_calendar_features,
    get_pd_name,
    alternate_blank_rows,
    coalesce,
    regexp_columns,
)

logging.getLogger("pymodules.pandas_utils").setLevel(logging.DEBUG)

EXAMPLE_DATAFRAME = pd.DataFrame(
    zip(np.arange(1, 100), np.arange(1, 100)),
    columns=pd.MultiIndex.from_arrays(
        (np.repeat("data", 2), ["column1", "column2"])
    ),
)

DATETIME_INDEXED_DATAFRAME = pd.DataFrame(
    zip(
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=100),
        np.arange(1, 100),
        np.arange(1, 100),
    ),
    columns=["datetime", "column1", "column2"],
).set_index("datetime")

DOUBLE_INDEXED_DATAFRAME = DATETIME_INDEXED_DATAFRAME.copy()
index_names = ["datetime1", "datetime2"]
DOUBLE_INDEXED_DATAFRAME.index = pd.MultiIndex.from_arrays(
    [DATETIME_INDEXED_DATAFRAME.index.values for _ in range(len(index_names))],
    names=index_names,
)


def test_format_series_name():
    """Returns series name as a string."""
    assert format_series_name(EXAMPLE_DATAFRAME.columns[0]) == "data column1"


def test_extract_calendar_features(caplog):
    assert list(
        extract_calendar_features(DATETIME_INDEXED_DATAFRAME).index.names
    ) == [
        "minute",
        "hour",
        "day",
        "dow",
        "weekday",
        "month",
        "date",
        "datetime",
    ]

    with pytest.raises(TypeError) as exc_info:
        extract_calendar_features(DATETIME_INDEXED_DATAFRAME.reset_index())
        assert exc_info.value.args[0] == "A DatetimeIndex level is required"

    # dual index
    extract_calendar_features(DOUBLE_INDEXED_DATAFRAME)
    assert (
        "More than one DatetimeIndex level is available, the first one: datetime1 will be used"
        in caplog.text
    )


def test_flatten_index():
    assert list(flatten_index(EXAMPLE_DATAFRAME).columns) == [
        "data column1",
        "data column2",
    ]

    assert (
        flatten_index(DOUBLE_INDEXED_DATAFRAME, axis="both").index.names[0]
        == "datetime1_datetime2"
    )


def test_get_pd_name():
    assert get_pd_name(EXAMPLE_DATAFRAME) == [
        ("data", "column1"),
        ("data", "column2"),
    ]
    assert get_pd_name(EXAMPLE_DATAFRAME.loc[:, ("data", "column1")]) == [
        ("data", "column1"),
    ]


def test_alternate_blank_rows():
    processed_df = alternate_blank_rows(EXAMPLE_DATAFRAME)
    assert len(processed_df) == len(EXAMPLE_DATAFRAME) * 2
    assert len(processed_df.dropna()) == len(EXAMPLE_DATAFRAME)


def test_coalesce():
    COALESCABLE_DATAFRAME = EXAMPLE_DATAFRAME.copy()
    COALESCABLE_DATAFRAME.iloc[np.random.randint(0, 99, 25), 0] = np.nan
    coalesced_df = coalesce(COALESCABLE_DATAFRAME)
    assert len(coalesced_df.columns) == 1 and list(coalesced_df.columns) == [
        ("data", "column1")
    ]

    COALESCABLE_DATAFRAME["additional_column"] = 10
    coalesced_df = coalesce(
        COALESCABLE_DATAFRAME,
        column_names=[("data", "column1"), ("data", "column2")],
        name=("data", "coalesced"),
    )
    assert len(coalesced_df.columns) == 2 and list(coalesced_df.columns) == [
        ("additional_column", ""),
        ("data", "coalesced"),
    ]

    with pytest.raises(ValueError):
        # Wrong name for the coalesced column
        coalesce(
            COALESCABLE_DATAFRAME,
            column_names=[("data", "column1"), ("data", "column2")],
            name="coalesced",
        )


def test_regexp_columns():
    assert regexp_columns("col", EXAMPLE_DATAFRAME) == [
        ("data", "column1"),
        ("data", "column2"),
    ]

    assert (
        len(regexp_columns("Col", EXAMPLE_DATAFRAME, case_sensitive=True)) == 0
    )
    assert len(regexp_columns("data", EXAMPLE_DATAFRAME, full_label=True)) == 0

    regexped_columns = regexp_columns(
        "data_column1", EXAMPLE_DATAFRAME, full_label=True
    )
    assert len(regexped_columns) == 1 and regexped_columns == [
        ("data", "column1")
    ]


def test_pandas_to_matlab(monkeypatch, caplog):
    def mock_scipy_io(*args):
        pass

    monkeypatch.setattr("scipy.io.savemat", mock_scipy_io)

    pandas_to_matlab(EXAMPLE_DATAFRAME, "matfile.mat")
    assert (
        "Saved DataFrame with [('index', ''), ('data', 'column1'), ('data', 'column2')] columns to matfile.mat matfile"
        in caplog.text
    )
