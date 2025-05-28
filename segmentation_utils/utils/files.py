from pathlib import Path
import itertools

import numpy as np


def keep_only_files(list_of_files_and_dirs):
    return sorted(filter(lambda x: x.is_file(), list_of_files_and_dirs))


def get_files_matching_extensions(folder, extensions):

    image_files = sorted(
        itertools.chain(
            *[keep_only_files((Path(folder).rglob("*" + ex))) for ex in extensions]
        )
    )
    return image_files


def get_matching_files(
    images_folder,
    labels_folder,
    image_extensions,
    label_extensions,
    ignore_substr_images="",
    ignore_substr_labels="",
):
    # If there is only one extension provided as a string, turn it into a one-length list
    if isinstance(image_extensions, str):
        image_extensions = [image_extensions]
    if isinstance(label_extensions, str):
        label_extensions = [label_extensions]

    image_files = get_files_matching_extensions(images_folder, image_extensions)
    label_files = get_files_matching_extensions(labels_folder, label_extensions)
    image_stems = [
        str(x.relative_to(images_folder).with_suffix("")).replace(
            ignore_substr_images, ""
        )
        for x in image_files
    ]
    label_stems = [
        str(x.relative_to(labels_folder).with_suffix("")).replace(
            ignore_substr_labels, ""
        )
        for x in label_files
    ]

    intersecting_stems = list(set(image_stems).intersection(set(label_stems)))
    image_files = np.array(
        [
            image_file
            for image_file, image_stem in zip(image_files, image_stems)
            if image_stem in intersecting_stems
        ]
    )
    label_files = np.array(
        [
            label_file
            for label_file, label_stem in zip(label_files, label_stems)
            if label_stem in intersecting_stems
        ]
    )
    return image_files, label_files
