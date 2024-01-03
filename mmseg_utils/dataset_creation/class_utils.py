import os
from pathlib import Path

import numpy as np
from imageio import imwrite
from skimage.transform import resize
from tqdm import tqdm

from mmseg_utils.dataset_creation.file_utils import read_npy_or_img


def remap_folder(
    input_folder, output_folder, remap, glob_str="*", output_img_size=None
):
    os.makedirs(output_folder, exist_ok=True)
    input_files = Path(input_folder).rglob(glob_str)
    input_files = list(filter(lambda x: x.is_file(), input_files))
    for input_file in tqdm(input_files):
        input_label = read_npy_or_img(input_file)
        output_label = remap_classes_bool_indexing(input_label, remap)
        if output_img_size is not None:
            output_label = resize(output_label, output_shape=output_img_size, order=0)

        output_file = Path(output_folder, input_file.relative_to(input_folder))
        if output_file.suffix == ".npy":
            output_file = output_file.with_suffix(".png")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_label = output_label.astype(np.uint8)
        imwrite(output_file, output_label)


def combine_classes(first_image, second_image, remap):
    """
    remap : np.ndarray
        An integer array where the element at the ith, jth location represents the class
        when a pixel in the first image has value i and the pixel in the second image has
        value j.
    """
    img_shape = first_image.shape
    first_image, second_image = [x.flatten() for x in (first_image, second_image)]
    remapped = remap[first_image, second_image]
    remapped = np.reshape(remapped, img_shape)
    return remapped


def remap_classes_bool_indexing(
    input_classes: np.array, remap: np.array, background_value: int = 255
):
    """Change indices based on input

    https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
    """
    if remap is None:
        return input_classes

    output = np.full_like(input_classes, dtype=np.uint8, fill_value=background_value)
    for i, v in enumerate(remap):
        mask = input_classes == i
        output[mask] = v
    return output


def remap_classes(input_classes: np.array, remap: np.array):
    """Change indices based on input

    https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
    """
    print("This function is slow, use remap_classes_bool_indexing")
    from_values = np.arange(len(remap))
    d = dict(zip(from_values, remap))

    input_shape = input_classes.shape
    input_classes = input_classes.flatten()

    out = [d[i] for i in input_classes]
    out = np.reshape(out, input_shape)
    return out
