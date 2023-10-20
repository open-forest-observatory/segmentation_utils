import pandas as pd
from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.draw import polygon as skimg_polygon
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil
from mmseg_utils.visualization.visualize_classes import visualize, show_colormaps
from mmseg_utils.dataset_creation.summary_statistics import compute_summary_statistics


from mmseg_utils.dataset_creation.file_utils import (
    write_cityscapes_file,
)
from mmseg_utils.config import (
    CLASS_MAP,
    IGNORE_INDEX,
    COLUMN_NAMES,
    PALETTE_MAP,
    CLASS_NAMES,
)
from mmseg_utils.dataset_creation.split_utils import get_is_train_array
from argparse import ArgumentParser

ANNOTATION_FILE = "/home/frc-ag-1/Downloads/oporto_2021_12_17_collect_1 (1).csv"
IMAGE_FOLDER = "/media/frc-ag-1/Elements/data/Safeforest_CMU_data_dvc/data/site_Oporto_clearing/2021_12_17/collect_1/processed_1/images/mapping_left"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-folder", default=IMAGE_FOLDER)
    parser.add_argument("--annotation-file", default=ANNOTATION_FILE)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--image-extension", default="jpg")
    parser.add_argument("--index-shift", type=int, default=0)
    parser.add_argument(
        "--dataset-identifier", choices=CLASS_MAP.keys(), default="safeforest23"
    )
    parser.add_argument("--write-unannotated", action="store_true")
    args = parser.parse_args()
    return args


def clip_locs_to_img(rr, cc, img_shape):
    # TODO make sure this order is correct
    valid_rr = np.logical_and(rr >= 0, rr < img_shape[1])
    valid_cc = np.logical_and(cc >= 0, cc < img_shape[0])
    valid = np.logical_and(valid_cc, valid_rr)
    return rr[valid], cc[valid]


def create_label_image(image_path, annotation_df, class_map, ignore_index=IGNORE_INDEX):
    image_file = image_path.parts[-1]
    matching_rows = annotation_df.loc[annotation_df["image_name"] == image_file]
    img = imread(image_path)
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
        label_img[cc, rr] = class_map[class_name]

    return img, label_img


def main(
    image_folder,
    annotation_file,
    output_folder,
    train_frac,
    skip_unannotated=True,
    dataset_identifier="",
    image_extension="jpg",
    seed=0,
    index_shift=0,
    ignore_index=IGNORE_INDEX,
):
    output_folder = Path(output_folder, dataset_identifier)
    output_train_img_dir = Path(output_folder, "img_dir", "train")
    output_train_ann_dir = Path(output_folder, "ann_dir", "train")

    os.makedirs(output_folder, exist_ok=True)
    class_map = CLASS_MAP[dataset_identifier]

    show_colormaps(
        PALETTE_MAP[dataset_identifier],
        CLASS_NAMES[dataset_identifier],
        #    savepath=Path(output_folder, "class_color_vis.png"),
    )

    image_paths = list(Path(image_folder).glob("*." + image_extension))
    annotation_df = pd.read_csv(annotation_file, sep=",", names=COLUMN_NAMES)

    num_total = len(image_paths)
    num_train = int(train_frac * num_total)
    is_train_array = get_is_train_array(num_total, num_train, seed=seed)
    index = 0
    for i, image_path in enumerate(tqdm(image_paths)):
        # Determine whether to use for training
        is_train = is_train_array[i]

        # Read the data and create label image
        img, label_img = create_label_image(
            image_path,
            annotation_df,
            class_map=class_map,
            ignore_index=ignore_index,
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
    shutil.copyfile(annotation_file, Path(output_folder, Path(annotation_file).name))

    visualize(
        seg_dir=Path(output_folder, "ann_dir", "train"),
        image_dir=Path(output_folder, "img_dir", "train"),
        output_dir=Path(output_folder, "train_vis"),
        palette_name=dataset_identifier,
    )
    compute_summary_statistics(
        images=output_train_img_dir,
        savepath=Path(output_folder, "summary_statistics.txt"),
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.image_folder,
        args.annotation_file,
        args.output_folder,
        args.train_frac,
        image_extension=args.image_extension,
        dataset_identifier=args.dataset_identifier,
        index_shift=args.index_shift,
        skip_unannotated=not args.write_unannotated,
    )
