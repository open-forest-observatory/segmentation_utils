import pandas as pd
from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.draw import polygon as skimg_polygon
import matplotlib.pyplot as plt
from tqdm import tqdm

from mmseg_utils.dataset_creation.file_utils import (
    get_files,
    link_cityscapes_file,
    write_cityscapes_file,
)
from mmseg_utils.dataset_creation.split_utils import get_is_train_array
from argparse import ArgumentParser

ANNOTATION_FILE = "/home/frc-ag-1/Downloads/oporto_2021_12_17_collect_1 (1).csv"
IMAGE_FOLDER = "/media/frc-ag-1/Elements/data/Safeforest_CMU_data_dvc/data/site_Oporto_clearing/2021_12_17/collect_1/processed_1/images/mapping_left"

CLASS_MAP = {
    "Fuel": 0,
    "Canopy": 1,
    "Background": 2,
    "Trunks": 3,
    "unknown": 2, 
}

# CLASS_MAP = {
#     "Dry Grass": 1,
#     "Green Grass": 2,
#     "Dry Shrubs": 3,
#     "Green Shrubs": 4,
#     "Canopy": 5,
#     "Wood Pieces": 6,
#     "Litterfall": 7,
#     "Timber Litter": 8,
#     "Live Trunks": 9,
#     "Bare Earth": 10,
#     "People": 11,
#     "Sky": 12,
#     "Blurry": 13,
#     "Obstacle": 14,
#     "Obstacles": 14,
#     "Drone": 15,
# }
# COLOR_MAP = {
#    "background": (0, 0, 0),
#    "fuel": (255, 0, 0),
#    "trunks": (255, 0, 255),
#    "canopy": (0, 255, 0),
# }

COLUMN_NAMES = (
    "column_ID",
    "image_name",
    "frame_ID",
    "bbox_tl_x",
    "bbox_tl_y",
    "bbox_br_x",
    "bbox_br_y",
    "det_or_len_confidence",
    "length",
    "class",
    "class_confidence",
    "polygon",
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-folder", default=IMAGE_FOLDER)
    parser.add_argument("--annotation-file", default=ANNOTATION_FILE)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--write-unannotated", action="store_true")
    args = parser.parse_args()
    return args


def clip_locs_to_img(rr, cc, img_shape):
    # TODO make sure this order is correct
    valid_rr = np.logical_and(rr >= 0, rr < img_shape[1])
    valid_cc = np.logical_and(cc >= 0, cc < img_shape[0])
    valid = np.logical_and(valid_cc, valid_rr)
    return rr[valid], cc[valid]


def create_label_image(image_path, annotation_df, create_vis_image=False, ignore_index=255):
    image_file = image_path.parts[-1]
    matching_rows = annotation_df.loc[annotation_df["image_name"] == image_file]
    img = imread(image_path)
    vis_img = img.copy()
    label_img = np.ones(img.shape[:2], dtype=np.uint8) * ignore_index
    for _, row in matching_rows.iterrows():
        try:
            polygon = np.array(row["polygon"][7:].split(" ")).astype(int)
        except TypeError:
            continue
        polygon = np.reshape(polygon, (int(polygon.shape[0] / 2), 2))

        rr, cc = skimg_polygon(polygon[:, 0], polygon[:, 1])
        rr, cc = clip_locs_to_img(rr, cc, label_img.shape[:2])
        class_name = row["class"]
        label_img[cc, rr] = CLASS_MAP[class_name]

        if create_vis_image:
            vis_img[cc, rr] = COLOR_MAP[class_name]
    return img, label_img, vis_img


def main(
    image_folder,
    annotation_file,
    output_folder,
    train_frac,
    skip_unannotated=True,
    seed=0,
    ignore_index=255,
):
    image_paths = list(Path(image_folder).glob("*.jpg"))
    annotation_df = pd.read_csv(annotation_file, sep=",", names=COLUMN_NAMES)

    num_total = len(image_paths)
    num_train = int(train_frac * num_total)
    is_train_array = get_is_train_array(num_total, num_train, seed=seed)
    index = 0
    for i, image_path in enumerate(tqdm(image_paths)):
        # Determine whether to use for training
        is_train = is_train_array[i]

        # Read the data and create label image
        img, label_img, _ = create_label_image(
            image_path, annotation_df, create_vis_image=False, ignore_index=ignore_index,
        )
        if skip_unannotated and np.all(label_img == ignore_index):
            continue

        write_cityscapes_file(
            img, output_folder, index, is_ann=False, is_train=is_train
        )
        write_cityscapes_file(
            label_img, output_folder, index, is_ann=True, is_train=is_train
        )
        index += 1


if __name__ == "__main__":
    args = parse_args()
    main(
        args.image_folder,
        args.annotation_file,
        args.output_folder,
        args.train_frac,
        skip_unannotated=not args.write_unannotated,
    )
