import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mmseg_utils.config import ANN_DIR, IMG_DIR, RGB_EXT, SEG_EXT, TRAIN_DIR, VAL_DIR
from mmseg_utils.dataset_creation.dataset_utils import process_dataset_images
from mmseg_utils.utils.files import get_matching_files
from mmseg_utils.visualization.visualize_classes import load_png_npy


def folder_to_cityscapes(
    images_folder,
    labels_folder,
    image_ext,
    label_ext,
    train_frac,
    val_frac,
    output_folder,
    remove_old,
    classes,
    vis_number,
    default_image_suffix=".JPG",
    default_label_suffix=".png",
):
    # TODO make this work with multiple file extensions, e.g. .jpg, .jpeg
    image_files, label_files = get_matching_files(
        images_folder,
        labels_folder,
        image_extensions=image_ext,
        label_extensions=label_ext,
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
    training_images = permuted_inds <= (n_valid * train_frac)
    # And the requested number of val are taken from the end
    # With the default training using 1.0 (all) of the data,
    # the validation images will be a subset of the training ones
    # the fact they are included at all is primarily because MMSeg requires at
    # least one validation image, and this can be somewhat helpful for metric
    # reporting, to make sure training is progressing. Though at this point
    # what is being reported is really training accuracy and not real val accuracy
    val_images = permuted_inds >= (n_valid * (1 - val_frac))

    img_train = Path(output_folder, IMG_DIR, TRAIN_DIR)
    img_val = Path(output_folder, IMG_DIR, VAL_DIR)
    ann_train = Path(output_folder, ANN_DIR, TRAIN_DIR)
    ann_val = Path(output_folder, ANN_DIR, VAL_DIR)

    if remove_old:
        [
            shutil.rmtree(x, ignore_errors=True)
            for x in (img_train, img_val, ann_train, ann_val)
        ]
    [os.makedirs(x, exist_ok=True) for x in (img_train, img_val, ann_train, ann_val)]

    for i in tqdm(
        range(n_valid), desc="Linking files into either train or test folders"
    ):
        stem = image_files[i].relative_to(images_folder)
        stem = str(stem.with_suffix("")).replace(os.path.sep, "_")

        image_suffix = (
            default_image_suffix
            if default_image_suffix is not None
            else image_files[i].suffix
        )
        label_suffix = (
            default_label_suffix
            if default_label_suffix is not None
            else label_files[i].suffix
        )

        train_image_destination = Path(img_train, f"{stem}{RGB_EXT}{image_suffix}")
        val_image_destination = Path(img_val, f"{stem}{RGB_EXT}{image_suffix}")

        train_ann_destination = Path(ann_train, f"{stem}{SEG_EXT}{label_suffix}")
        val_ann_destination = Path(ann_val, f"{stem}{SEG_EXT}{label_suffix}")

        # This is weird because an image may be part of 0, 1, or both sets
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
        class_names=classes,
        output_folder=output_folder,
        vis_number=vis_number,
    )
