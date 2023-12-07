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

from mmseg_utils.config import (
    COLUMN_NAMES,
    IGNORE_INDEX,
    MATPLOTLIB_PALLETE,
    RGB_EXT,
    SEG_EXT,
)
from mmseg_utils.dataset_creation.file_utils import (
    link_cityscapes_file,
    write_cityscapes_file,
)
from mmseg_utils.dataset_creation.mmseg_config import create_new_config
from mmseg_utils.dataset_creation.split_utils import get_is_train_array
from mmseg_utils.dataset_creation.summary_statistics import compute_summary_statistics
from mmseg_utils.visualization.visualize_classes import show_colormaps, visualize


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-folder")
    parser.add_argument("--annotation-file")
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--image-extension", default="jpg")
    parser.add_argument("--write-unannotated", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()
    return args


def create_label_image(image_path, annotation_df, class_map, ignore_index=IGNORE_INDEX):
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
        # Convert to (I assume) i, j points
        polygon = np.reshape(polygon, (int(polygon.shape[0] / 2), 2))

        polygon_mask = polygon2mask(polygon=polygon, image_shape=img_shape)
        class_ID = class_map[row["class"]]

        class_mask = label_mask_dict[class_ID]
        # Add the new label
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
    vis=False,
    write_config=False,
):
    output_train_img_dir = Path(output_folder, "img_dir", "train")

    os.makedirs(output_folder, exist_ok=True)
    meta_file = Path(Path(annotation_file).parent, "meta.json")

    with open(meta_file, "r") as infile:
        metadata = json.load(infile)

    class_map = {x: i for i, x in enumerate(metadata["customTypeStyling"].keys())}
    class_map["unknown"] = ignore_index
    class_names = list(class_map.keys())
    class_names.pop(class_names.index("unknown"))

    show_colormaps(
        MATPLOTLIB_PALLETE,
        class_names=class_names,
        savepath=Path(output_folder, "class_color_vis.png"),
    )

    image_paths = list(Path(image_folder).glob("*." + image_extension))
    annotation_df = pd.read_csv(annotation_file, sep=",", names=COLUMN_NAMES)

    num_total = len(image_paths)
    num_train = int(train_frac * num_total)
    is_train_array = get_is_train_array(num_total, num_train, seed=seed)

    shutil.copyfile(annotation_file, Path(output_folder, Path(annotation_file).name))
    valid_index = 0
    for i, image_path in enumerate(tqdm(image_paths, desc="Converting data")):
        # Determine whether to use for training
        # TODO this should be computed using the number of valid ones, not pre-computed
        is_train = is_train_array[i]

        # Read the data and create label image
        label_img = create_label_image(
            image_path,
            annotation_df,
            class_map=class_map,
            ignore_index=ignore_index,
        )
        if skip_unannotated and np.all(label_img == ignore_index):
            continue

        link_cityscapes_file(
            image_path,
            output_folder,
            valid_index,
            is_ann=False,
            is_train=is_train,
            exist_ok=True,
        )
        write_cityscapes_file(
            label_img, output_folder, valid_index, is_ann=True, is_train=is_train
        )

        # Increament number of valid images
        valid_index += 1

    if write_config:
        mean, std = compute_summary_statistics(
            images=output_train_img_dir,
            savepath=Path(output_folder, "summary_statistics.txt"),
        )
        output_config = Path(output_folder, Path(output_folder).stem + ".py")
        create_new_config(
            "configs/cityscapes_forests.py",
            output_config_file=output_config,
            mean=mean,
            std=std,
            classes=class_names,
            data_root=output_folder,
        )

    if vis:
        visualize(
            seg_dir=Path(output_folder, "ann_dir", "train"),
            image_dir=Path(output_folder, "img_dir", "train"),
            output_dir=Path(output_folder, "train_vis"),
            ignore_substr_images_for_matching=RGB_EXT,
            ignore_substr_labels_for_matching=SEG_EXT,
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
