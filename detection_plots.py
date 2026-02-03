from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional
import os
import sys
from skimage import exposure


def enhance_contrast(
    image: np.ndarray, min_intensity=0.2, max_intensity=0.99
) -> np.ndarray:
    vmin, vmax = np.percentile(image, [min_intensity, max_intensity])
    clipped_data = exposure.rescale_intensity(
        image, in_range=(vmin, vmax), out_range=np.float32
    )
    return clipped_data


def equalise_hist(image: np.ndarray) -> np.ndarray:
    return exposure.equalize_hist(image)


def equalise_adaptive_hist(image: np.ndarray) -> np.ndarray:
    return exposure.equalize_adapthist(image, clip_limit=0.03)


def create_detection_plots(
    mask: np.ndarray,
    image: np.ndarray,
    save_path: Path,
    granule_cmap: str = "tab20",
    image_cmap: str | LinearSegmentedColormap = "inferno",
    randomised_mask: np.ndarray | None = None,
):
    granule_cmap: plt.Colormap = plt.get_cmap(granule_cmap)
    granule_cmap.set_bad((0, 0, 0, 0))

    im_data = image
    im_shape = im_data.shape
    aspect = im_shape[1] / im_shape[0]

    titles = ["Detected", "Original", "Overlay"]
    if randomised_mask is not None:
        titles.append("Randomised Mask")

    fig, axs = create_axes(len(titles), aspect=aspect, col_wrap=2)

    masked_granules = np.ma.masked_equal(mask, 0)
    axs[0].imshow(masked_granules, cmap=granule_cmap, interpolation="none")
    axs[1].imshow(
        image,
        cmap=image_cmap,
    )

    axs[2].imshow(
        image,
        cmap=image_cmap,
    )
    axs[2].imshow(masked_granules, cmap=granule_cmap, interpolation="none", alpha=0.3)

    if randomised_mask is not None:
        random_cmap: plt.Colormap = plt.get_cmap("viridis")
        random_cmap.set_under((0, 0, 0, 0))
        axs[3].imshow(
            randomised_mask,
            cmap=random_cmap,
            interpolation="none",
            vmax=1.0,
            vmin=0.005,
        )

    tick_spacing = 128

    for ax, title in zip(axs, titles):
        ax.set_title(title)

        x_ticks = np.arange(0, im_shape[1], step=tick_spacing)
        ax.set_xticks(x_ticks)
        y_ticks = np.arange(0, im_shape[0], step=tick_spacing)
        ax.set_yticks(y_ticks)

        ax.grid(ls="--", lw=0.2)

    save_figure_and_trim(save_path, dpi=330)


def create_axes(
    n_axes: int = 1,
    col_wrap: int = 4,
    axes_height: Optional[float] = 4.0,
    fig_width: Optional[float] = None,
    aspect: float = 1.0,
    squeeze=True,
    projection: Optional[str] = None,
    **subplot_kw,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create a figure and sub-axes, while specifing the size of the sub figures.

    This emulated the behaviour of the seaborn facet grid, where we give a size
    for the individual figures rather than the overall plot.

    The remaining axes are blanked using plt.axis('off')

    Only one of ``axes_height`` or ``fig_width`` should be not None as these
    options conflict. To maintain backwards compatibility, ``axes_height`` will
    override ``fig_width``.

    Parameters
    ----------
    n_axes: int (default=1)
        Number of sub-axes to create
    col_wrap: int (default=4)
        How many columns in the figure, next axes will go into new rows.
    axes_height: float or None
        Height of the axes, in inches by default
    fig_width: float or None
        Total width of the figure, conflicts with axes height
    aspect: float
        Width/Height ratio of the sub-axes
    squeeze: bool
        If True, extra dimensions are squeezed out from the returned axes array
        typically, this is used to remove the singleton dimension when only one
        axes is created.
    projection: str
        The projection type of the axes. Can also be provided as in ``subplot_kw``.
    subplot_kw: dict
        Additional keyword arguments to pass to plt.subplots

    Returns
    -------
    fig, [axes]
        The axes are returned in a flat array, [0,...,n_axes-1]

    """
    n_cols = min(n_axes, col_wrap)
    n_rows = int(np.ceil(n_axes / n_cols))
    n_axes_blank = n_cols * n_rows - n_axes

    if fig_width is None and axes_height is None:
        raise ValueError()

    match (fig_width, axes_height):
        case (None, None):
            raise ValueError(
                "At least one of fig_width and fig_height must be non-None"
            )
        case (None, ax_height):
            axes_width = ax_height * aspect
            fig_width = n_cols * axes_width
            fig_height = n_rows * ax_height
        case (fig_width, _):
            axes_width = fig_width / n_cols
            axes_height = axes_width / aspect
            fig_height = axes_height * n_rows

    if projection is not None:
        if not subplot_kw:
            subplot_kw = dict(subplot_kw=dict())
        else:
            subplot_kw["subplot_kw"] = dict()
        subplot_kw["subplot_kw"]["projection"] = projection

    fig, axs = plt.subplots(
        ncols=n_cols,
        nrows=n_rows,
        figsize=(fig_width, fig_height),
        **subplot_kw,
    )  # type: ignore

    # Keep the axes returned in a 1d array
    if n_rows > 1:
        axs = axs.flatten()

    # Blank the axis at the end of the list if we create more than specified
    for blank_axes in range(n_axes_blank):
        axs[-(blank_axes + 1)].axis("off")

    if not squeeze and n_axes == 1:
        axs = [axs]

    return fig, axs


def hide_axis_labels(ax):
    """Given an axis, remove the text and spacing used in the tick marks."""
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis("off")


def save_figure_and_trim(
    save_name: Path,
    args=None,
    additional_metadata=None,
    padding=0.15,
    tl_padding=1.08,
    fig=None,
    despine=True,
    dpi=330,
    background=None,
    close=True,
    transparent=False,
):
    """Save the figure with metadata and crop the boundaries."""
    # Use the most recent figure if None is provided
    if fig is None:
        fig = plt.gcf()

    # Remove the right and top axis
    axs = fig.get_axes()
    if despine:
        [despine_axis(ax) for ax in axs]

    if Path(save_name).suffix == ".png":
        metadata = {
            "Exp:Creating Script": f"{Path(sys.argv[0]).resolve()}",
            "Exp:Working Dir": f"{os.getcwd()}",
        }

        # Provide the command line arguments
        if args is not None:
            for key, value in vars(args).items():
                metadata[f"Exp:arg-{key}"] = f"{str(value)}"

        # Add user provided keywords
        if additional_metadata is not None:
            metadata.update(additional_metadata)
    else:
        metadata = None

    plot_kwargs = {}
    if padding:
        plot_kwargs = dict(bbox_inches="tight", pad_inches=padding)

    if tl_padding is not None:
        fig.tight_layout(pad=tl_padding)
    fig.savefig(
        save_name,
        dpi=dpi,
        **plot_kwargs,
        metadata=metadata,
        facecolor=background,
        transparent=transparent,
    )
    if close:
        plt.close(fig=fig)


def despine_axis(ax):
    """Remove the top and right axis.

    This emulates seaborn.despine, but doesn't require the library.

    Will fail silently if the spines don't exist, for example in a polar plot.
    """
    try:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    except KeyError:
        pass


microscope_green = LinearSegmentedColormap.from_list(
    "microscope_green",
    [f"#00{i}{i}00" for i in ["0", "8", "b", "d", "e", "f"]],
)

microscope_blue = LinearSegmentedColormap.from_list(
    "microscope_blue",
    [f"#0000{i}{i}" for i in ["0", "8", "b", "d", "e", "f"]],
)

grey_green = LinearSegmentedColormap.from_list(
    "grey_green",
    ["black", "#666666", "#449944", "#44FF44"],
)

green_white = LinearSegmentedColormap.from_list(
    "green_white", ["black", "#00FF00", "white"]
)

microscope_red = LinearSegmentedColormap.from_list(
    "microscope_red",
    [f"#{i}{i}0000" for i in ["0", "8", "b", "d", "e", "f"]],
)

microscope_magenta = LinearSegmentedColormap.from_list(
    "microscope_magenta",
    [f"#{i}{i}00{i}{i}" for i in ["0", "8", "b", "d", "e", "f"]],
)

microscope_cyan = LinearSegmentedColormap.from_list(
    "microscope_cyan",
    [f"#00{i}{i}{i}{i}" for i in ["0", "8", "b", "d", "e", "f"]],
)

microscope_yellow = LinearSegmentedColormap.from_list(
    "microscope_yellow",
    [f"#{i}{i}{i}{i}00" for i in ["0", "8", "b", "d", "e", "f"]],
)
