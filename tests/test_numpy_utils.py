import logging

import numpy as np
import pytest

from pymodules.numpy_utils import (
    moving_average,
    shift,
    base_round,
    window_slide,
    consecutive_steps,
    find_nearest,
)

logging.getLogger("pymodules.numpy_utils").setLevel(logging.DEBUG)


def test_moving_average():
    assert moving_average(np.arange(10)).tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_shift():
    with pytest.raises(TypeError):
        shift(np.arange(10), 1).tolist()

    assert np.isnan(shift(np.arange(10).astype("float"), 1)[0])
    assert np.flatnonzero(shift(np.arange(10), 3, fill_value=0)).size == 6


def test_base_round():
    assert base_round(5, 10) == 0  # Round to the nearest even number
    assert base_round(6, 10) == 10
    assert base_round(6.6, 2) == 6
    assert base_round(6.6, 2.2) == 6.6000000000000005


def test_window_slide():
    assert window_slide(np.arange(5, 15)).tolist() == [
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        [8, 9, 10],
        [9, 10, 11],
        [10, 11, 12],
        [11, 12, 13],
        [12, 13, 14],
    ]

    assert window_slide(np.arange(5, 15), stepsize=2, width=1).tolist() == [
        [5],
        [7],
        [9],
        [11],
        [13],
    ]

    assert len(window_slide(np.arange(5, 15), stepsize=1, width=11).tolist()) == 0


def test_consecutive_steps():
    assert len(consecutive_steps(np.arange(5, 15))) == 10
    assert all(
        [
            a.tolist() == b.tolist()
            for a, b in zip(
                consecutive_steps(np.array([2, 5, 5, 6, 15]), stepsize=4, direction="smaller"),
                [
                    np.array([2]),
                    np.array([5]),
                    np.array([5]),
                    np.array([6, 15]),
                ],
            )
        ]
    )
    assert all(
        [
            a.tolist() == b.tolist()
            for a, b in zip(
                consecutive_steps(
                    np.array([2, 5, 5, 6, 15]),
                    stepsize=9,
                    direction="exact",
                ),
                [np.array([2, 5, 5, 6]), np.array([15])],
            )
        ]
    )


def test_find_nearest():
    assert find_nearest(np.arange(5, 15), 14.5) == 14
    assert find_nearest(np.arange(5, 15), 0) == 5
    assert find_nearest(np.arange(5, 15), -5) == 5
