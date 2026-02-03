import random
from fontTools.misc.iterTools import zip_longest
import enum
import logging
from pathlib import Path

import os
from rich.logging import RichHandler


import numpy as np
import pandas as pd
from cellpose import models
from cellpose.io import imread
from skimage import exposure, measure, filters, morphology
from skimage.measure._regionprops import RegionProperties

import detection_plots as dp


def main(input_dir: Path, save_dir: Path, pre_enhance_image: bool = False):
    setup_logger()

    image_paths = _parse_input_images(input_dir)
    cell_df = []
    for image_path in image_paths:
        try:
            df_part = process_image(
                image_path, save_dir, pre_enhance_image=pre_enhance_image
            )
            cell_df.append(df_part)
        except ValueError as e:
            logging.error(f"Unable to process image, {image_path}: {e}")

    cell_df: pd.DataFrame = pd.concat(cell_df, ignore_index=True)
    print(cell_df)
    cell_df.to_csv(save_dir / "stats.csv")


def process_image(
    image_path: Path, save_dir: Path, pre_enhance_image: bool = True
) -> pd.DataFrame:
    model = models.CellposeModel(gpu=True)
    image = read_image(image_path)
    logging.debug(f"Starting segmentation {image_path}")

    enhancement_mode = EnhancementMethod.ADAPT_HIST
    enhanced_image = enhancement_mode(image)

    if pre_enhance_image:
        # Normalisation seems to break with the enhanced image
        image_to_segment = enhanced_image
        normalisation_args = dict(normalize=True)
    else:
        image_to_segment = image
        normalisation_args = dict(lowhigh=[1.0, 99.0], normalize=True)

    # Matching the normalize values in the GUI, can probably do better than this though.
    masks, flows, styles = model.eval(
        image_to_segment,
        diameter=60,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        normalize=normalisation_args,
    )
    logging.info(f"Found {masks.max()} masks")

    save_path = save_dir / f"{image_path.stem}.png"

    randomised_mask = randomise_mask(masks)
    boundary_mask = convert_mask_to_boundary(randomised_mask)

    dp.create_detection_plots(
        boundary_mask=boundary_mask,
        filled_mask=randomised_mask,
        image=enhanced_image,
        save_path=save_path,
        image_cmap="gist_yarg",
    )

    image_props = {"name": image_path.stem, "count": masks.max()}
    return pd.DataFrame(
        data=image_props,
        index=[
            0,
        ],
    )


def read_image(image_path: Path) -> np.ndarray:
    image = imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    return image


class EnhancementMethod(enum.IntEnum):
    NONE = enum.auto()
    HIST = enum.auto()
    ADAPT_HIST = enum.auto()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        match self:
            case EnhancementMethod.NONE:
                return image
            case EnhancementMethod.HIST:
                return dp.equalise_hist(image)
            case EnhancementMethod.ADAPT_HIST:
                return dp.equalise_adaptive_hist(image)

            case _:
                raise ValueError(f"{self} not supported for image enhancement.")


def convert_mask_to_boundary(labelled_masks: np.ndarray) -> np.ndarray:
    props: list[RegionProperties] = measure.regionprops(labelled_masks)
    base_arr = np.zeros_like(labelled_masks, dtype="int64")

    footprint = morphology.disk(radius=7)

    for prop in props:
        local_mask = prop.image

        shrunk_area = morphology.binary_erosion(local_mask, footprint)
        ring = np.logical_xor(local_mask, shrunk_area)

        base_arr[prop.slice] += ring * prop.label

    return base_arr


def randomise_mask(labelled_masks: np.ndarray, low_value: float = 0.01) -> np.ndarray:
    """ "Replace the labels on the mask with a random number.

    This helps break up the grouping of the colours in the plot.
    A ``low_value`` can be provided, where the random values are in the range (low_value, 1.0),
    this allows a bit of an easier distinction between the background and the labels.
    """
    # Seeded so that it'll at least be identical given the same input
    rng = np.random.default_rng(42)

    old_labels = np.arange(1, labelled_masks.max())
    mixed_labels = old_labels.copy()
    rng.shuffle(mixed_labels)

    randomised_mask = np.zeros_like(labelled_masks)
    for old_label, new_label in zip(old_labels, mixed_labels):
        label_region = labelled_masks == old_label
        randomised_mask[label_region] = new_label

    return randomised_mask


def _parse_input_images(image_path: Path, extension: str = "*.jpg") -> list[Path]:
    if image_path.is_file():
        logging.info(f"Given image {image_path} directly")
        return [image_path]

    if image_path.is_dir():
        images = list(image_path.glob(extension))
        n_images = len(images)
        if n_images == 0:
            raise SystemExit(f"No granules found in directory {image_path}")

        logging.info(f"Found {n_images} images in {image_path}")
        return images

    raise SystemExit(f"Given path {image_path} is not a file or directory")


def setup_logger():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    return logging.getLogger("rich_logger")


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "sample-data"
    input_dir = data_dir / "faz-images/Phase"
    save_dir = data_dir / "faz-images/phase-out-enhanced"

    main(input_dir, save_dir)
