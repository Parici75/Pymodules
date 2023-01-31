"""Numpy utilities."""

from typing import Any, Sequence, List

import numpy as np


def moving_average(x: np.ndarray, sliding_window: int = 3) -> np.ndarray:
    """From https://stackoverflow.com/a/14314054"""
    cumulative_sum = np.cumsum(x, dtype="float")
    cumulative_sum[sliding_window:] = (
        cumulative_sum[sliding_window:] - cumulative_sum[:-sliding_window]
    )
    return cumulative_sum[sliding_window - 1:] / sliding_window


def shift(
    arr: np.ndarray, offset: int, fill_value: Any = np.nan
) -> np.ndarray:
    """Shifts a numpy array."""
    if type(fill_value) != arr.dtype:
        raise TypeError(f"Fill value: {fill_value} is not of type {arr.dtype}")

    if offset == 0:
        return arr

    result = np.empty_like(arr)
    if offset > 0:
        result[:offset] = fill_value
        result[offset:] = arr[:-offset]
    elif offset < 0:
        result[offset:] = fill_value
        result[:offset] = arr[-offset:]

    return result


def base_round(x: float, base: float = 10.0) -> float:
    """Rounds a number to the closest base."""
    return base * np.round(x / base)


def window_slide(
    data: Sequence, stepsize: int = 1, width: int = 3
) -> np.ndarray:
    """Generates a list of subarrays obtained with a sliding window of stepsize and width.

    See https://stackoverflow.com/a/40085052/4696032
    """
    array = np.asarray(data)  # Convert to array
    n_rows = ((array.size - width) // stepsize) + 1
    return array[stepsize * np.arange(n_rows)[:, None] + np.arange(width)]


def consecutive_steps(
    data: Sequence, stepsize: int = 1, direction: str = "bigger"
) -> List[np.ndarray]:
    """Cut arrays where difference between consecutive elements is equal to or bigger than stepsize.

    Adapted from https://stackoverflow.com/a/7353335
    """

    if direction == "bigger":
        return np.split(data, np.where(np.diff(data) >= stepsize)[0] + 1)

    if direction == "smaller":
        return np.split(data, np.where(np.diff(data) < stepsize)[0] + 1)

    if direction == "exact":
        return np.split(data, np.where(np.diff(data) == stepsize)[0] + 1)

    raise ValueError(
        f"Invalid direction argument: '{direction}' is not one of ['bigger', 'smaller', 'exact']"
    )


def find_nearest(data: Sequence, value: float) -> float:
    """Finds element closest to value in a ndarray."""
    array = np.asarray(data)
    idx = np.abs(array - value).argmin()

    return array.flat[idx]
