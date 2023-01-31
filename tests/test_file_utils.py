import logging
import os
from unittest.mock import patch, mock_open

import pytest

from pymodules.file_utils import glob_re, process_filename, write_dict_to_csv

logging.getLogger("pymodules.file_utils").setLevel(logging.DEBUG)

EXAMPLE_DIRECTORY_LIST = [
    "C://mock_directory/mock_file1.mock",
    "C://mock_directory/mock_file25.mock",
]


def test_glob_re():
    assert len(glob_re("mock_directory", EXAMPLE_DIRECTORY_LIST)) == 2
    assert len(glob_re("mock_file1", EXAMPLE_DIRECTORY_LIST)) == 1
    assert len(glob_re(r"mock_file\d.mock", EXAMPLE_DIRECTORY_LIST)) == 1
    assert len(glob_re(r"mock_\b", EXAMPLE_DIRECTORY_LIST)) == 0
    assert len(glob_re(r".mock", EXAMPLE_DIRECTORY_LIST)) == 2


def test_process_filename():
    assert process_filename("mock filename.py") == "Mock_filename.py"
    assert process_filename("mock@filename/") == "Mock_filename_"


def test_write_dict_to_csv():
    valid_dictionary_list = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 3, "b": 4, "c": 5},
    ]
    invalid_dictionary_list = [{"a": 1, "b": 2, "c": 3}, {"a": 3, "b": 4}]
    mock_file = os.path.join("mock_directory", "exports")
    with patch("pymodules.file_utils.open", mock_open()) as mocked_file:
        write_dict_to_csv(valid_dictionary_list, mock_file)

        # assert if open context was used 3 times to call a method
        mocked_file.assert_called_with(mock_file, "w", encoding="utf-8")
        mocked_file.call_count == 3

        # assert that the last write method call is as expected
        mocked_file().write.assert_called_with(
            f"{','.join([str(i) for i in valid_dictionary_list[1].values()])}\r\n"
        )

        # value error when attempting to write "non iso" dictionaries.
        with pytest.raises(ValueError):
            write_dict_to_csv(invalid_dictionary_list, mock_file)
