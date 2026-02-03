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


def main(input_dir: Path, save_dir: Path):
    setup_logger()

    image_paths = _parse_input_images(input_dir)
    for image_path in image_paths:
        try:
            process_image(image_path, save_dir)
        except ValueError as e:
            logging.error(f"Unable to process image, {image_path}: {e}")

    pass


def process_image(image_path: Path, save_dir: Path) -> pd.DataFrame:
    model = models.CellposeModel(gpu=True)
    image = read_image(image_path)
    logging.debug(f"Starting segmentation {image_path}")

    # Matching the normalize values in the gui, can probably do better than this though.
    normalisation_args = dict(lowhigh=[1.0, 99.0], normalize=True)
    masks, flows, styles = model.eval(
        image,
        diameter=60,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        normalize=normalisation_args,
    )
    logging.info(f"Found {masks.max()} masks")

    save_path = save_dir / f"{image_path.stem}.png"
    enhancement_mode = EnhancementMethod.ADAPT_HIST
    enhanced_image = enhancement_mode(image)

    randomised_mask = randomise_mask(masks)
    boundary_mask = convert_mask_to_boundary(randomised_mask)

    dp.create_detection_plots(
        boundary_mask,
        enhanced_image,
        save_path,
        image_cmap="gist_yarg",
        randomised_mask=randomised_mask,
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

    randomised_mask = np.zeros_like(labelled_masks, dtype=float)
    for i in range(1, labelled_masks.max()):
        label_region = labelled_masks == i
        randomised_mask[label_region] = rng.uniform(low=low_value, high=1.0)
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
    # input_dir = data_dir / "faz-images/Phase"
    save_dir = data_dir / "faz-images/PhaseOut"

    input_dir = Path(
        "/home/carl/scratch/cellpose-test/sample-data/faz-images/Phase/20251216_Faz-SGs__A1_1_00d00h00m.jpg"
    )

    main(input_dir, save_dir)
