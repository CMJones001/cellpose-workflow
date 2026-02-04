#!/usr/bin/env python
from properties_table import create_summary
import argparse
import enum
import logging
import os
from pathlib import Path
from time import perf_counter

import ncolor
import numpy as np
import pandas as pd
from cellpose import models
from cellpose.io import imread
from rich.logging import RichHandler
from skimage import measure, morphology
from skimage.measure._regionprops import RegionProperties

import detection_plots as dp


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


def main(
    input_dir: Path,
    save_dir: Path,
    pre_enhance_image: bool = False,
    enhancement_method: EnhancementMethod = EnhancementMethod.ADAPT_HIST,
    four_colour: bool = False,
):
    logger = setup_logger()
    start_time = perf_counter()

    image_paths = _parse_input_images(input_dir)
    cell_df = []
    for image_path in image_paths:
        try:
            df_part = process_image(
                image_path,
                save_dir,
                pre_enhance_image=pre_enhance_image,
                enhancement_method=enhancement_method,
                four_colour=four_colour,
            )
            cell_df.append(df_part)
        except ValueError as e:
            logging.error(f"Unable to process image, {image_path}: {e}")
        except NoMasksFound as _:
            logging.warning(f"No cells found in {image_path}")

    end_time = perf_counter()

    cell_df: pd.DataFrame = pd.concat(cell_df, ignore_index=True)
    cell_df.to_csv(save_dir / "stats.csv")

    summary_df = create_summary(cell_df, image_paths)
    summary_df.to_csv(save_dir / "summary.csv")

    if logger.isEnabledFor(logging.INFO):
        duration = end_time - start_time
        n_images = len(image_paths)

        def to_minutes(time_sec: int | float) -> str:
            if time_sec < 60:
                return f"{time_sec:.1f}"
            minutes = time_sec // 60
            sec_remainder = time_sec % 60
            return f"{minutes}:{sec_remainder:02d}"

        logging.info(
            f"Time taken {to_minutes(duration)} seconds ({to_minutes(duration / n_images)} s / image)"
        )


def process_image(
    image_path: Path,
    save_dir: Path,
    pre_enhance_image: bool = True,
    enhancement_method: EnhancementMethod = EnhancementMethod.ADAPT_HIST,
    four_colour: bool = False,
) -> pd.DataFrame:
    model = models.CellposeModel(gpu=True)
    image = read_image(image_path)
    logging.info(f"Starting segmentation {image_path}")

    enhancement_mode = EnhancementMethod.ADAPT_HIST
    enhanced_image = enhancement_mode(image)

    if pre_enhance_image:
        image_to_segment = enhanced_image
        # Normalisation seems to break with the enhanced image
        normalisation_args = dict(normalize=True)
    else:
        image_to_segment = image
        # Matching the normalize values in the GUI, can probably do better than this though.
        normalisation_args = dict(lowhigh=[1.0, 99.0], normalize=True)

    masks, flows, styles = model.eval(
        image_to_segment,
        diameter=60,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        normalize=normalisation_args,
    )

    n_masks = masks.max()
    if n_masks == 0:
        raise NoMasksFound(f"No masks found in {image_path}")
    logging.info(f"Found {n_masks} masks")

    save_path = save_dir / f"{image_path.stem}.png"

    if four_colour:
        randomised_mask = ncolor.label(masks)
        mask_cmap = "Paired"
    else:
        randomised_mask = randomise_mask(masks)
        mask_cmap = "tab20"

    boundary_mask = convert_filled_mask_to_boundary(randomised_mask)

    dp.create_detection_plots(
        boundary_mask=boundary_mask,
        filled_mask=randomised_mask,
        image=enhanced_image,
        save_path=save_path,
        mask_cmap=mask_cmap,
        image_cmap="gist_yarg",
    )

    image_props = get_granule_properties_table(masks, image_to_segment)
    image_props["im_name"] = image_path.stem
    image_props["count"] = masks.max()

    return image_props


def read_image(image_path: Path) -> np.ndarray:
    image = imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    return image


def get_granule_properties_table(
    labelled_image: np.ndarray, image: np.ndarray
) -> pd.DataFrame:
    # This doesn't need the intensity ``image``, but nice to have incase we want to add something
    # else.
    granule_df = pd.DataFrame(
        measure.regionprops_table(
            labelled_image,
            image,
            properties=[
                "label",
                "area",
                "major_axis_length",
                "equivalent_diameter_area",
                "perimeter_crofton",
                "perimeter",
                "eccentricity",
            ],
        )
    )
    # type: ignore
    # Derive some extra columns that aren't provided by skimage
    granule_df["circularity_equiv"] = (
        granule_df["major_axis_length"] / granule_df["equivalent_diameter_area"]
    )
    granule_df["circularity_crofton"] = (
        granule_df["perimeter"] / granule_df["perimeter_crofton"]
    )

    # Drop some columns that were used for computations
    granule_df = granule_df.drop(
        columns=["major_axis_length", "equivalent_diameter_area", "perimeter_crofton"]
    )
    return granule_df


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


def convert_filled_mask_to_boundary(
    filled_mask: np.ndarray, boundary_thickness: int = 7
) -> np.ndarray:
    """Convert a filled, labelled image into a labelled set of boundaries."""
    props: list[RegionProperties] = measure.regionprops(filled_mask)
    boundary_mask = np.zeros_like(filled_mask, dtype="int64")

    footprint = morphology.disk(radius=boundary_thickness)

    for prop in props:
        local_mask = prop.image

        shrunk_area = morphology.binary_erosion(local_mask, footprint)
        ring = np.logical_xor(local_mask, shrunk_area)

        # As we do an erosion, the ring will always be smaller than the original mask,
        # so this is okay, we'd have to handle overlaps more carefully otherwise
        boundary_mask[prop.slice] += ring * prop.label

    return boundary_mask


def randomise_mask(labelled_masks: np.ndarray) -> np.ndarray:
    """Shuffle the labels on the mask.

    If nearby cells have similar numbers, then they'll likely share the same colour in the plots,
    making the segmentation appear worse. Shuffling the labels helps with this, but can't entirely
    eliminate the problem.
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
    """Return a list of images from a file or directory.

    If image_path is a file, return a one-element list containing that file.
    If image_path is a directory, return all files in the directory matching the given glob
    pattern (extension).

    Quits the program if no files are found.
    """
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


def setup_logger() -> logging.Logger:
    """Add some prettiness to the logging output, also read the ``LOG_LEVEL`` envvar."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    return logging.getLogger("rich_logger")


def parse_args():
    parser = argparse.ArgumentParser(description="Image enhancement tool")
    parser.add_argument("input_dir", type=Path, help="Directory of input images")
    parser.add_argument("save_dir", type=Path, help="Directory to save enhanced images")
    parser.add_argument(
        "--pre_enhance_image",
        action="store_true",
        help="Perform image enhancements before segmentation?",
    )

    parser.add_argument(
        "--enhancement_method",
        type=str,
        choices=[e.name for e in EnhancementMethod],
        default=EnhancementMethod.ADAPT_HIST.name,
        help="Choice of enhancement method (NONE, HIST, ADAPT_HIST)",
    )
    parser.add_argument(
        "-f",
        "--four_colour",
        action="store_true",
        help="Use four colouring on the segmentation.",
    )

    return parser.parse_args()


def run_cli():
    args = parse_args()
    main(
        input_dir,
        save_dir,
        args.pre_enhance_image,
        enhancement_method=args.enhancment_method,
    )


class NoMasksFound(Exception):
    pass


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "sample-data"
    input_dir = data_dir / "faz-images/Phase"
    save_dir = data_dir / "faz-images/phase-out-enhanced"

    main(
        input_dir,
        save_dir,
        four_colour=True,
        enhancement_method=EnhancementMethod.ADAPT_HIST,
        pre_enhance_image=True,
    )
