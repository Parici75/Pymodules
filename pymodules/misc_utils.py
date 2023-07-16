"""Miscellaneous utilities."""
from __future__ import annotations

from itertools import zip_longest
from typing import Any, Dict, Generator, Iterable


def grouper(n: int, iterable: Iterable, fill_value: Any = None) -> "zip_longest":
    """Collects data into fixed-length chunks or blocks.

    example:
        grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    """
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fill_value, *args)


def pairwise(iterable: Iterable) -> Generator[Any, None, None]:
    """Generates pairs of adjacent elements in an iterable."""
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield a, b
        a = b


def invert_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Invert mapping of a dictionary.

    If values are not unique, then append keys to a list.
    """

    inv_map: Dict[str, Any] = {}
    if len(set(mapping.values())) < len(mapping.values()):
        # Values are not unique
        for k, v in mapping.items():
            inv_map.setdefault(v, []).append(k)
    else:
        inv_map = {v: k for k, v in mapping.items()}

    return inv_map


def flatten(items: Iterable) -> Generator[Any, None, None]:
    """Yield items from any nested iterable
    see https://stackoverflow.com/a/40857703/4696032
    """

    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x
