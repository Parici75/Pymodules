import logging

import matplotlib
import numpy as np
from matplotlib.pyplot import get_cmap

from pymodules.matplotlib_utils import (
    get_diverging_cmap,
    array_to_cmap,
    to_grayscale_cmap,
)

logging.getLogger("pymodules.matplotlib_utils").setLevel(logging.DEBUG)


def test_get_diverging_cmap(caplog):
    assert (
        get_diverging_cmap(top_cmap="Oranges", bottom_cmap="Blues_r")(
            np.arange(128)
        ).tolist()
        == get_cmap("Blues_r", lut=128)(np.arange(128)).tolist()
    )


def test_array_to_cmap():
    cmap = "jet"
    # Test input array
    array = np.array([0.2, 0.4, 0.6, 0.8])
    # Test output from function
    output = array_to_cmap(array, cmap, normalize=False)
    # Test expected output
    expected_output = get_cmap(cmap)(array)
    # Assert that output matches expected output
    assert (output == expected_output).all()

    # The `normalize` keyword is equivalent to passing a vector of a linearly spaced intervals between 0 and 1
    assert (
        array_to_cmap(np.arange(10), cmap=cmap, normalize=True)
        == array_to_cmap(np.linspace(0, 1, 10), cmap=cmap)
    ).all()


def test_grayscale_cmap():
    grayscale_cmap = to_grayscale_cmap("jet")
    # check if the returned object is of the correct type
    assert isinstance(
        grayscale_cmap, matplotlib.colors.LinearSegmentedColormap
    )
    # check if the colors are grayscale
    colors = grayscale_cmap(np.arange(grayscale_cmap.N))
    assert all(
        (colors[:, 0] == colors[:, i]).all()
        for i in np.arange(1, colors.shape[1] - 1)
    )
