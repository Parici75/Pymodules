import logging

import numpy as np

from pymodules.misc_utils import (
    grouper,
    pairwise,
    invert_mapping,
    flatten,
)

logging.getLogger("pymodules.misc_utils").setLevel(logging.DEBUG)


def test_grouper():
    assert list(grouper(2, ["a", "b"])) == [("a", "b")]
    assert list(grouper(2, {"a": 1, "b": 2, "c": 3})) == [
        ("a", "b"),
        ("c", None),
    ]
    assert list(grouper(2, np.arange(7), fill_value="no_number")) == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, "no_number"),
    ]


def test_pairwise():
    assert list(pairwise({"a": 1, "b": 2, "c": 3, "d": 4})) == [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
    ]

    assert list(pairwise(np.arange(-5, 5))) == [
        (-5, -4),
        (-4, -3),
        (-3, -2),
        (-2, -1),
        (-1, 0),
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]


def test_invert_mapping():
    assert invert_mapping({"a": 1, "b": 2, "c": 3, "d": 4}) == {
        1: "a",
        2: "b",
        3: "c",
        4: "d",
    }
    assert invert_mapping({"a": 1, "b": 2, "c": 3, "d": 4, "e": 4}) == {
        1: ["a"],
        2: ["b"],
        3: ["c"],
        4: ["d", "e"],
    }


def test_flatten():
    assert list(flatten([[[[1], [2, 3]], 4], [5]])) == [1, 2, 3, 4, 5]
