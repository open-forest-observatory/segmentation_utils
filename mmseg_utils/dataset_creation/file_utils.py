import os
from pathlib import Path

import numpy as np
from imageio import imread
from ubelt import ensuredir, symlink

from mmseg_utils.config import ANN_DIR, IMG_DIR, RGB_EXT, SEG_EXT, TRAIN_DIR, VAL_DIR
from mmseg_utils.dataset_creation.img_utils import imwrite_skimage


def pad_filename(filename: Path, start_index=None, end_index=None, pad_length: int = 6):
    """
    filename:
        The filename to pad. Can be relative or absolute path
    pad_length:
        How digits it should have

    returns:
        The padded filename, at the same depth it was originally
    """
    parent = filename.parent
    stem = int(filename.stem[start_index:end_index])
    extension = filename.suffix
    formatted_stem = f"{stem:0{pad_length}d}"
    output = Path(parent, formatted_stem + extension)
    return output


def get_files(
    folder: Path, pattern: str, sort=True, require_dir=False, require_file=False
):
    """
    args:

    folder:
        The input folder
    pattern:
        The file to search for within that folder
    sort:
        Return the sorted generator
    require_file:
        Only return files
    require_dir:
        Only return directories

    returns:
        a list or generator of Path 's
    """
    if require_dir and require_file:
        raise ValueError(
            "Both require_dir and require_file set. Nothing satisfies both."
        )

    files = Path(folder).glob(pattern)
    if sort:
        files = sorted(files)

    if require_dir:
        files = [f for f in files if f.is_dir()]

    if require_file:
        files = [f for f in files if f.is_file()]

    return files


def generate_output_file(output_folder, index, is_ann, is_train, suffix=".png"):
    """
    output_folder:
    index:
    is_ann:
    is_train:
    """
    output_sub_folder = Path(
        output_folder,
        ANN_DIR if is_ann else IMG_DIR,
        TRAIN_DIR if is_train else VAL_DIR,
    )
    ensuredir(output_sub_folder, mode=0o0755)
    filename = f"{index:06d}{SEG_EXT if is_ann else RGB_EXT}{suffix}"
    output_filepath = Path(output_sub_folder, filename)
    return output_filepath


def read_npy_or_img(filename):
    filename = Path(filename)
    if filename.suffix == ".npy":
        return np.load(filename)
    return imread(filename)


def write_cityscapes_file(
    img: np.array, output_folder: Path, index: int, is_ann: bool, is_train: bool
):
    output_filepath = generate_output_file(output_folder, index, is_ann, is_train)
    img = img.astype(np.uint8)
    imwrite_skimage(str(output_filepath), img)


def link_cityscapes_file(
    img_path, output_folder, index, is_ann, is_train, exist_ok=False
):
    output_filepath = generate_output_file(
        output_folder, index, is_ann, is_train, suffix=Path(img_path).suffix
    )
    # Don't link if already there, coult be dangerous
    if os.path.isfile(output_filepath) and exist_ok:
        return

    symlink(img_path, output_filepath)


def ensure_dir_normal_bits(folder):
    ensuredir(folder, mode=0o0755)


def get_train_val_test(
    input_rgb_dir,
    input_seg_dir,
    test_frac,
    train_frac,
    extension="*.png",
    shuffle_test=True,
):
    rgb_files = np.asarray(sorted(input_rgb_dir.glob(extension)))
    seg_files = np.asarray(sorted(input_seg_dir.glob(extension)))

    all_files = rgb_files.tolist() + seg_files.tolist()
    common_root = os.path.commonpath(all_files)

    rgb_files = np.asarray([x.relative_to(common_root) for x in rgb_files])
    seg_files = np.asarray([x.relative_to(common_root) for x in seg_files])
    num_train = int((1 - test_frac) * rgb_files.shape[0])
    if shuffle_test:
        train_val_inds = np.zeros((rgb_files.shape[0],), dtype=bool)
        train_val_locs = np.argsort(np.random.uniform(size=(len(rgb_files),)))[
            :num_train
        ]
        train_val_inds[train_val_locs] = True
        test_inds = np.logical_not(train_val_inds)
        rgb_files_test = rgb_files[test_inds]
        seg_files_test = seg_files[test_inds]
        rgb_files_train_val = rgb_files[train_val_inds]
        seg_files_train_val = seg_files[train_val_inds]
    else:
        rgb_files_test = rgb_files[num_train:]
        seg_files_test = seg_files[num_train:]
        rgb_files_train_val = rgb_files[:num_train]
        seg_files_train_val = seg_files[:num_train]

    random_vals = np.random.uniform(size=(num_train,))
    train_ids = random_vals < train_frac
    val_ids = np.logical_not(train_ids)
    return (
        (rgb_files_train_val[train_ids], rgb_files_train_val[val_ids], rgb_files_test),
        (seg_files_train_val[train_ids], seg_files_train_val[val_ids], seg_files_test),
        common_root,
    )
