"""File utilities."""
from __future__ import annotations

import csv
import logging
import os
import pickle
import re
from datetime import datetime
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


def glob_re(pattern: str, directory_list: List[str]) -> List[str]:
    """Filters glob output with a regex pattern."""
    return list(filter(re.compile(pattern).search, directory_list))


def save_obj(object: Any, filename: str) -> None:
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(object, f)
    logger.info(f"{filename} saved !")


def load_obj(filename: str) -> Any:
    with open(filename + ".pkl", "rb") as f:
        object = pickle.load(f)
    logger.info(f"{filename} loaded !")
    return object


def process_filename(file_title_bits: Sequence) -> str:
    """Joins string bits, replacing white space with underscores and capitalizing first letter."""
    if not isinstance(file_title_bits, str):
        filename = "_".join([i.replace(" ", "_") for i in file_title_bits if i])
    else:
        filename = file_title_bits.replace(" ", "_")

    # Remove forbidden characters
    filename = re.sub(r"[^\w_. -]", "_", filename)

    return "".join([filename[0].upper(), filename[1:]])


def cd(newdir: str) -> None:
    """Commands to a directory."""
    if os.path.isdir(newdir):  # newdir is a directory
        os.chdir(newdir)
    else:  # newdir is a file, find the directory which contains it
        os.chdir(os.path.dirname(newdir))

    logger.info(f"Working directory: {os.getcwd()}")


def mkdir(newdir: str) -> str:
    """Creates a directory and returns its string representation."""
    newdir_path = os.path.abspath(newdir)
    if not os.path.isdir(newdir_path):
        os.makedirs(newdir_path, exist_ok=True)
        logger.info(f"Created directory: {newdir_path}")

    return newdir_path


def write_dict_to_csv(dictionary_list: List[Dict[str, Any]], output_file: str) -> None:
    """Exports a list of dictionaries to a csv."""

    fieldnames = [record.keys() for record in dictionary_list]
    if fieldnames.count(fieldnames[0]) != len(fieldnames):
        raise ValueError(f"Dictionary keys are not all identical")

    try:
        with open(os.path.normpath(output_file), "w", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(dictionary_list[0].keys()))
            writer.writeheader()
            for data in dictionary_list:
                writer.writerow(data)
    except IOError as exc:
        logger.error(f"I/O error: {exc}")


def build_timestamped_dir(root_folder: str | None = None) -> str:
    if root_folder is None:
        root_folder = os.getcwd()
    timestamped_dir = os.path.join(root_folder, datetime.now().strftime("%Y%m%d-%H%M%S"))
    return timestamped_dir
