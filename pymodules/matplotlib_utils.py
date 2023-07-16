"""Graphics and Matplotlib utilities."""
from __future__ import annotations

import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap

logger = logging.getLogger(__name__)


def get_diverging_cmap(
    top_cmap: matplotlib.colors.Colormap | str = "Oranges",
    bottom_cmap: matplotlib.colors.Colormap | str = "Blues_r",
    n_color: int = 256,
) -> matplotlib.colors.ListedColormap:
    """Constructs a diverging colormap.

    Args:
        top_cmap:
            The matplotlib colormap reference for the upper range of values.
        bottom_cmap:
            The matplotlib colormap reference for the lower range of values.
        n_color:
            The number of colors of the entire colormap

    Returns:
        A diverging Matplotlib colormap object.
    """
    top_cmap = plt.get_cmap(top_cmap, int(n_color / 2))
    bottom_cmap = plt.get_cmap(bottom_cmap, int(n_color / 2))

    diverging_colors = np.vstack(
        (
            bottom_cmap(np.linspace(0, 1, int(n_color / 2))),
            top_cmap(np.linspace(0, 1, int(n_color / 2))),
        )
    )
    diverging_cmap = ListedColormap(
        diverging_colors, name="_".join((top_cmap.name, bottom_cmap.name))
    )

    return diverging_cmap


def corr_imshow(
    corr: np.ndarray, ax: matplotlib.pyplot.axes = None, cbar: bool = True
) -> matplotlib.image.AxesImage:
    """Plots a matrix of correlation using a colorscale appropriately centered on 0."""

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Plot the image with the correctly centered colormap using TwoSlopeNorm
    im = ax.imshow(corr, norm=TwoSlopeNorm(), cmap=get_diverging_cmap())
    if cbar:
        fig.colorbar(im)

    return im


def array_to_cmap(
    array: np.ndarray,
    cmap: matplotlib.colors.Colormap | str,
    normalize: bool = True,
) -> np.ndarray:
    """Returns a rgba color array from an array of numeric values linearly mapped on a colormap."""

    cmap = plt.get_cmap(cmap)

    if normalize:
        norm = Normalize(vmin=min(array), vmax=max(array))
        return cmap(norm(array))

    return cmap(array)


def to_grayscale_cmap(
    cmap: matplotlib.colors.Colormap | str,
) -> matplotlib.colors.LinearSegmentedColormap:
    """Returns a grayscale version of the given colormap."""

    cmap = plt.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived grayscale luminance (see http://alienryderflex.com/hsp.html)
    rgb_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, rgb_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return matplotlib.colors.LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)


def plot_colormap(cmap: matplotlib.colors.Colormap | str) -> None:
    """Plots a colormap alongside its grayscale equivalent."""

    cmap = plt.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    grayscale_colors = to_grayscale_cmap(np.arange(to_grayscale_cmap(cmap).N))

    fig, ax = plt.subplots(2, figsize=(6, 2), subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale_colors], extent=[0, 10, 0, 1])


def remove_spines(
    spine_option: str | None = "nobox",
    ax: matplotlib.pyplot.Axes | None = None,
) -> matplotlib.pyplot.Axes:
    """Removes matplotlib box as in Matlab set(box, visible: off)."""

    if ax is None:
        ax = plt.gca()

    if spine_option == "naked":
        for spine in ax.spines.values():
            spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

    elif spine_option == "nobox":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    elif spine_option == "bottom":
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks([])
    else:
        raise ValueError(
            f"Invalid spine_option argument: {spine_option} is not one of ['naked', 'nobox', 'bottom']"
        )

    return ax


def show_plot() -> None:
    # Trick to show the plot and not be blocked in interactive mode
    plt.ion()
    plt.show()
    plt.pause(0.001)


class PlotSaver(object):
    """Wrapper class to export matplotlib Figures.

    Args:
        output_dir:
            The output directory.

    """

    def __init__(self, output_dir: str = os.getcwd()):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.output_dir = os.path.abspath(output_dir)

    def export_fig(
        self,
        plot_name: str = "plot",
        fig: matplotlib.pyplot.Figure | None = None,
    ) -> None:
        """Exports the matplotlib figure"""

        if fig is None:
            figure = plt.gcf()

        plot_path = os.path.join(self.output_dir, plot_name)

        figure.tight_layout()
        figure.savefig(plot_path, bbox_inches="tight")
        logger.info(f"Saved figure {plot_name} to {plot_path}")
        plt.close(figure)
