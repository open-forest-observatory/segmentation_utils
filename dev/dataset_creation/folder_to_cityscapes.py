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
    parser.add_argument("--train-frac", type=float, default=1.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--vis-number", type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # Find the files that match between the two folder trees, modulo image extensions
    image_files, label_files = get_matching_files(
        args.images_folder,
        args.labels_folder,
        args.image_ext,
        args.label_ext,
        ignore_substr_images=RGB_EXT,
        ignore_substr_labels=SEG_EXT,
    )

    # Check for images with no rendered label
    valid_labels = np.array(
        list(
            map(
                lambda x: np.any(load_png_npy(x) != 255),
                tqdm(label_files, desc="Checking for completely null images"),
            )
        )
    )

    # Remove invalid images
    image_files = image_files[valid_labels]
    label_files = label_files[valid_labels]
    # and compute how many are valid
    n_valid = len(image_files)

    # Split into train and (potentially overlapping) validation
    # This can be thought of randomly ordering the images
    permuted_inds = np.random.permutation(n_valid)
    # The requested number of training images are taken from the beginning
    training_images = permuted_inds <= (n_valid * args.train_frac)
    # And the requested number of val are taken from the end
    # With the default training using 1.0 (all) of the data,
    # the validation images will be a subset of the training ones
    # the fact they are included at all is primarily because MMSeg requires at
    # least one validation image, and this can be somewhat helpful for metric
    # reporting, to make sure training is progressing. Though at this point
    # what is being reported is really training accuracy and not real val accuracy
    val_images = permuted_inds >= (n_valid * (1 - args.val_frac))

    img_train = Path(args.output_folder, IMG_DIR, TRAIN_DIR)
    img_val = Path(args.output_folder, IMG_DIR, VAL_DIR)
    ann_train = Path(args.output_folder, ANN_DIR, TRAIN_DIR)
    ann_val = Path(args.output_folder, ANN_DIR, VAL_DIR)

    if args.remove_old:
        [
            shutil.rmtree(x, ignore_errors=True)
            for x in (img_train, img_val, ann_train, ann_val)
        ]
    [os.makedirs(x, exist_ok=True) for x in (img_train, img_val, ann_train, ann_val)]

    for i in tqdm(
        range(n_valid), desc="Linking files into either train or test folders"
    ):
        stem = image_files[i].relative_to(args.images_folder)
        stem = str(stem.with_suffix("")).replace(os.path.sep, "_")

        train_image_destination = Path(
            img_train,
            f"{stem}{RGB_EXT}{image_files[i].suffix}",
        )
        val_image_destination = Path(img_val, f"{stem}{RGB_EXT}{image_files[i].suffix}")
        train_ann_destination = Path(
            ann_train,
            f"{stem}{SEG_EXT}{label_files[i].suffix}",
        )
        val_ann_destination = Path(
            ann_val,
            f"{stem}{SEG_EXT}{label_files[i].suffix}",
        )

        if training_images[i]:
            os.symlink(
                image_files[i],
                train_image_destination,
            )
            os.symlink(label_files[i], train_ann_destination)
        if val_images[i]:
            os.symlink(image_files[i], val_image_destination)
            os.symlink(label_files[i], val_ann_destination)

    process_dataset_images(
        training_images_folder=img_train,
        class_names=args.classes,
        output_folder=args.output_folder,
        vis_number=args.vis_number,
    )
