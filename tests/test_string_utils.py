import logging

from pymodules.string_utils import (
    is_in_alphabetical_order,
    is_numeric,
    is_list_of_strings,
    dict_to_string,
    replace_strings,
)

logging.getLogger("pymodules.string_utils").setLevel(logging.DEBUG)


def test_is_in_alphabetical_order():
    assert is_in_alphabetical_order(["a", "b", "c"])
    assert not is_in_alphabetical_order(["b", "a", "c"])
    assert is_in_alphabetical_order(["a", "b", "c"])
    assert not is_in_alphabetical_order(["a", "b", "c", "0"])


def test_is_numeric():
    assert not is_numeric("zc*/5")
    assert is_numeric("58")
    assert not is_numeric("")


def test_is_list_of_strings():
    assert is_list_of_strings(["ab"])
    assert not is_list_of_strings(["b", 2])
    assert not is_list_of_strings(["ab", ("c",)])
    assert not is_list_of_strings("ab")
    assert not is_list_of_strings({})


def test_dict_to_string():
    assert dict_to_string({"a": 34, "b": 45}) == "a=34_b=45"
    assert dict_to_string({"2": "a", "3": "z"}) == "2=a_3=z"
    assert dict_to_string({"2": ["a", "b"], "3": "z"}) == "2=['a', 'b']_3=z"


def test_replace_strings():
    assert replace_strings("abc", {"a": "1", "bc": "2"}) == "12"
    assert replace_strings("abc", {"abcd": "1"}) == "abc"
    assert replace_strings("abc", {"a": "1", "ab": "2"}) == "2c"
