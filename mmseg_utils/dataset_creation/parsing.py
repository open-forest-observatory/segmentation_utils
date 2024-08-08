import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from imageio import imwrite
from skimage.draw import polygon2mask
from skimage.io import imread
from tqdm import tqdm
import typing

from mmseg_utils.config import COLUMN_NAMES, IGNORE_INDEX


def check_if_image(file):
    try:
        imread(file)
        return True
    except (ValueError, FileNotFoundError):
        return False


def parse_viame_annotation(
    image_path: Path,
    annotation_df: pd.DataFrame,
    class_map: dict,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """_summary_

    Args:
        image_path (Path): Full path to the image to generate label for
        annotation_df (pd.DataFrame): The dataframe with all annotations
        class_map (dict): Mapping from string class name to integer ID
        ignore_index (int, optional): This is the value for unlabeled pixels. Defaults to IGNORE_INDEX.

    Returns:
        np.ndarray: Labeled array (h, w) with integer labels
    """
    # Get the filename from the full path
    image_name = image_path.parts[-1]
    # Get the rows from the annotation dataframe matching this file
    matching_rows = annotation_df.loc[annotation_df["image_name"] == image_name]
    # Read in the image, but just to get the shape
    # TODO see if this could be accelerated by reading exif, probably not worth it though
    img_shape = imread(image_path).shape[:2]

    label_mask_dict = defaultdict(lambda: np.zeros(img_shape, dtype=bool))

    # Iterate over the rows in the dataframe for that image
    for _, row in matching_rows.iterrows():
        # TODO figure out why this is required, I assume the polygon may not be present
        try:
            polygon = np.array(row["polygon"][7:].split(" ")).astype(int)
        except TypeError:
            continue
        # Convert to i, j points
        polygon = np.flip(np.reshape(polygon, (int(polygon.shape[0] / 2), 2)), axis=1)
        # Create a mask for the area inside the polygon
        polygon_mask = polygon2mask(polygon=polygon, image_shape=img_shape)
        # Get the class ID for this polygon
        class_ID = class_map[row["class"]]
        # Extract the current mask for that class
        class_mask = label_mask_dict[class_ID]
        # Update the label for that class
        label_mask_dict[class_ID] = np.logical_or(class_mask, polygon_mask)

    # Pre-populate the label image with the null value
    label_img = np.ones(img_shape, dtype=np.uint8) * ignore_index

    # Sort in decending order of number of values
    # This means that infrequent classes will still be present even if they overlap with a more frequent class
    sorted_masks = sorted(
        label_mask_dict.items(), key=lambda x: np.sum(x[1]), reverse=True
    )
    # Populate the output
    for label_ID, mask in sorted_masks:
        # Don't overwrite valid information with unknown
        if label_ID == ignore_index:
            pass
        label_img[mask] = label_ID

    return label_img


def parse_viame_annotations_dataset(
    image_folder: Path,
    annotation_file: Path,
    output_folder: Path,
    class_map: typing.Union[Path, None] = None,
    ignore_index: int = IGNORE_INDEX,
):
    # Make output directory
    os.makedirs(output_folder, exist_ok=True)
    # Copy the annotation file to the output folder
    shutil.copyfile(annotation_file, Path(output_folder, Path(annotation_file).name))

    # Read in the annotations
    annotation_df = pd.read_csv(annotation_file, sep=",", names=COLUMN_NAMES)

    # TODO allow the class map to be passed in, either directly or as a path to a json
    if class_map is None:
        meta_file = Path(Path(annotation_file).parent, "meta.json")

        with open(meta_file, "r") as infile:
            metadata = json.load(infile)

        class_map = {x: i for i, x in enumerate(metadata["customTypeStyling"].keys())}
        class_map["unknown"] = ignore_index
        class_names = list(class_map.keys())
        class_names.pop(class_names.index("unknown"))
    elif isinstance(class_map, (Path, str)):
        with open(class_map, "r") as infile:
            class_map = json.load(infile)
    elif not isinstance(class_map, dict):
        raise ValueError("Not a valid class map")

    # List all of the images in the folder
    all_paths = list(Path(image_folder).glob("*"))
    # See if the file is a valid image
    image_paths = [f for f in all_paths if check_if_image(f)]

    # Iterate over images
    for image_path in tqdm((image_paths), desc="Converting data"):
        # Read the data and create label image
        label_img = parse_viame_annotation(
            image_path,
            annotation_df,
            class_map=class_map,
            ignore_index=ignore_index,
        )
        # The output file is the input filename in the output folder
        output_file = Path(output_folder, image_path.name)
        # Write the output file
        imwrite(output_file, label_img)
