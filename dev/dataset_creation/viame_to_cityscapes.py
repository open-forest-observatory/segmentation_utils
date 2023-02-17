import pandas as pd
from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.draw import polygon as skimg_polygon
import matplotlib.pyplot as plt

from mmseg_utils.dataset_creation.file_utils import (
    get_files,
    link_cityscapes_file,
    write_cityscapes_file,
)
from mmseg_utils.dataset_creation.split_utils import get_is_train_array
from argparse import ArgumentParser

ANNOTATION_FILE = "/home/frc-ag-1/Downloads/oporto_2021_12_17_collect_1 (1).csv"
IMAGE_FOLDER = "/media/frc-ag-1/Elements/data/Safeforest_CMU_data_dvc/data/site_Oporto_clearing/2021_12_17/collect_1/processed_1/images/mapping_left"

CLASS_MAP = {"background": 1, "fuel": 2, "trunks": 3, "canopy": 4}
COLOR_MAP = {
    "background": (0, 0, 0),
    "fuel": (255, 0, 0),
    "trunks": (255, 0, 255),
    "canopy": (0, 255, 0),
}

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
    args = parser.parse_args()
    return args


def create_label_image(image_path, annotation_df, create_vis_image=False):
    image_file = image_path.parts[-1]
    matching_rows = annotation_df.loc[annotation_df["image_name"] == image_file]
    img = imread(image_path)
    vis_img = img.copy()
    label_img = np.zeros(img.shape[:2], dtype=np.uint8)
    for _, row in matching_rows.iterrows():
        polygon = np.array(row["polygon"][7:].split(" ")).astype(int)
        polygon = np.reshape(polygon, (int(polygon.shape[0] / 2), 2))

        rr, cc = skimg_polygon(polygon[:, 0], polygon[:, 1])
        class_name = row["class"]
        label_img[cc, rr] = CLASS_MAP[class_name]

        if create_vis_image:
            vis_img[cc, rr] = COLOR_MAP[class_name]
    return img, label_img, vis_img


def main(image_folder, annotation_file):
    image_paths = Path(image_folder).glob("*.png")
    annotation_df = pd.read_csv(annotation_file, sep=",", names=COLUMN_NAMES,)

    for image_path in image_paths:
        img, label_img, vis_img = create_label_image(
            image_path, annotation_df, create_vis_image=True
        )
        plt.imshow(label_img)
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args.image_folder, args.annotation_file)
