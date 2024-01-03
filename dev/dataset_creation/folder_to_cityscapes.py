import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mmseg_utils.config import ANN_DIR, IMG_DIR, RGB_EXT, SEG_EXT, TRAIN_DIR, VAL_DIR
from mmseg_utils.dataset_creation.dataset_utils import process_dataset_images
from mmseg_utils.utils.files import get_matching_files
from mmseg_utils.visualization.visualize_classes import load_png_npy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-folder")
    parser.add_argument("--labels-folder")
    parser.add_argument("--output-folder")
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--image-ext", default="JPG")
    parser.add_argument("--label-ext", default="png")
    parser.add_argument("--remove-old", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--vis-stride", type=int, default=20)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    image_files, label_files = get_matching_files(
        args.images_folder,
        args.labels_folder,
        args.image_ext,
        args.label_ext,
        ignore_substr_images=RGB_EXT,
        ignore_substr_labels=SEG_EXT,
    )

    valid_labels = np.array(
        list(
            map(
                lambda x: np.any(load_png_npy(x) != 255),
                tqdm(label_files, desc="Checking for completely null images"),
            )
        )
    )

    image_files = image_files[valid_labels]
    label_files = label_files[valid_labels]
    n_valid = len(image_files)
    training_images = np.random.permutation(n_valid) < (n_valid * args.train_frac)

    IMG_TRAIN = Path(args.output_folder, IMG_DIR, TRAIN_DIR)
    IMG_VAL = Path(args.output_folder, IMG_DIR, VAL_DIR)
    ANN_TRAIN = Path(args.output_folder, ANN_DIR, TRAIN_DIR)
    ANN_VAL = Path(args.output_folder, ANN_DIR, VAL_DIR)

    if args.remove_old:
        [
            shutil.rmtree(x, ignore_errors=True)
            for x in (IMG_TRAIN, IMG_VAL, ANN_TRAIN, ANN_VAL)
        ]
    [os.makedirs(x, exist_ok=True) for x in (IMG_TRAIN, IMG_VAL, ANN_TRAIN, ANN_VAL)]

    for i in tqdm(
        range(n_valid), desc="Linking files into either train or test folders"
    ):
        stem = image_files[i].relative_to(args.images_folder)
        stem = str(stem.with_suffix("")).replace(os.path.sep, "_")

        os.symlink(
            image_files[i],
            Path(
                (IMG_TRAIN if training_images[i] else IMG_VAL),
                f"{stem}{RGB_EXT}{image_files[i].suffix}",
            ),
        )
        os.symlink(
            label_files[i],
            Path(
                (ANN_TRAIN if training_images[i] else ANN_VAL),
                f"{stem}{SEG_EXT}{label_files[i].suffix}",
            ),
        )

    process_dataset_images(
        training_images_folder=IMG_TRAIN,
        class_names=args.classes,
        output_folder=args.output_folder,
        vis_stride=args.vis_stride,
    )
