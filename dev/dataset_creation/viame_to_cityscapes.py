import json
import os
import shutil
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.draw import polygon2mask
from skimage.io import imread
from tqdm import tqdm

from mmseg_utils.config import COLUMN_NAMES, IGNORE_INDEX
from mmseg_utils.dataset_creation.dataset_utils import process_dataset_images
from mmseg_utils.dataset_creation.file_utils import (
    link_cityscapes_file,
    write_cityscapes_file,
)
from mmseg_utils.dataset_creation.split_utils import get_is_train_array


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-folder")
    parser.add_argument("--annotation-file")
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--image-extension", default="jpg")
    parser.add_argument("--write-unannotated", action="store_true")
    args = parser.parse_args()
    return args


def create_label_image(
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


def main(
    image_folder,
    annotation_file,
    output_folder,
    train_frac,
    skip_unannotated=True,
    image_extension="jpg",
    seed=0,
    ignore_index=IGNORE_INDEX,
    class_map=None,
):
    # Make output directory
    output_train_img_dir = Path(output_folder, "img_dir", "train")
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

    # List all of the images in the folder
    image_paths = list(Path(image_folder).glob("*." + image_extension))
    # Mark which images are for training vs validation
    num_images = len(image_paths)
    is_train_array = get_is_train_array(
        num_total=num_images, num_train=int(train_frac * num_images), seed=seed
    )

    valid_index = 0
    for is_train, image_path in tqdm(
        zip(is_train_array, image_paths), desc="Converting data"
    ):
        # Read the data and create label image
        label_img = create_label_image(
            image_path,
            annotation_df,
            class_map=class_map,
            ignore_index=ignore_index,
        )
        # If no pixels are annotated, skip
        if skip_unannotated and np.all(label_img == ignore_index):
            continue
        # Link the image into the output structure
        link_cityscapes_file(
            image_path,
            output_folder,
            valid_index,
            is_ann=False,
            is_train=is_train,
            exist_ok=True,
        )
        # Write the label in the cityscapes format
        write_cityscapes_file(
            label_img, output_folder, valid_index, is_ann=True, is_train=is_train
        )

        # Increament number of valid images
        valid_index += 1

    # Compute the summary statistics
    process_dataset_images(
        training_images_folder=output_train_img_dir,
        class_names=class_names,
        output_folder=output_folder,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.image_folder,
        args.annotation_file,
        args.output_folder,
        args.train_frac,
        image_extension=args.image_extension,
        skip_unannotated=not args.write_unannotated,
        vis=args.vis,
        write_config=args.write_config,
    )
