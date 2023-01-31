"""String utilities."""
import re
from typing import Any, Dict, Sequence


def is_in_alphabetical_order(word: Sequence) -> bool:
    """Returns True if a sequence of strings follows alphabetical order."""

    return all([word[i + 1] >= word[i] for i in range(len(word) - 1)])


def is_numeric(string: str) -> bool:
    """Check if a string contains only numeric character."""
    return re.match("^[0-9\.]+$", string) is not None


def is_list_of_strings(sequence: Sequence) -> bool:
    """See https://stackoverflow.com/questions/18495098/how-to-check-if-an-object-is-a-list-of-strings."""

    return (
        bool(sequence)
        and isinstance(sequence, list)
        and all(isinstance(elem, str) for elem in sequence)
    )


def dict_to_string(dictionary: Dict[str, Any], sep: str = "_") -> str:
    """Generates a string with key/value pairs of a dictionary."""
    kv_list = []
    for kv in dictionary.items():
        kv_list.append("=".join(map(str, kv)))

    return sep.join(kv_list)


def replace_strings(string: str, replacements_pairs: Dict[str, str]) -> str:
    """Substitutes multiple substring recursively using key/value mapping in repls_pairs.

    See https://stackoverflow.com/a/58814507
    """
    while len(replacements_pairs) > 0:
        string = replace_strings(
            string.replace(*replacements_pairs.popitem()), replacements_pairs
        )
    return string
