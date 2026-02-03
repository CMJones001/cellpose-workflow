"""
Calculate properties of the granules from the image and summarising them.
"""

import numpy as np
import pandas as pd
from skimage import measure
from pathlib import Path


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


def create_summary(granule_df: pd.DataFrame, image_paths: list[Path]) -> pd.DataFrame:
    """Condense the full data table to one row per image."""

    # Setting the names as categories means that all names will be included even if they
    # have no granules.
    image_names = [i.stem for i in image_paths]
    granule_df["im_name"] = pd.Categorical(
        granule_df["im_name"], categories=image_names
    )

    # Slightly cursed way of getting the count in one column and the mean in the rest without
    # having to deal with multi-columns
    columns_of_interest: list[str] = [
        "label",
        "area",
        "perimeter",
        "circularity_crofton",
        "circularity_equiv",
        "eccentricity",
    ]
    function_mapping = {n: "mean" for n in columns_of_interest}
    # There was a change here with pandas 3.0, this used to allow `im_name` as a key, but not any more...
    # doesn't really matter, as we're just counting
    function_mapping["label"] = "count"

    summary_table = granule_df.pivot_table(
        index="im_name",
        values=columns_of_interest,
        aggfunc=function_mapping,
        observed=False,
    ).reset_index()
    summary_table = summary_table.rename(columns={"label": "count"})
    return summary_table
