import argparse
import enum
import itertools
import os
import shutil
import typing
from pathlib import Path

from mmseg_utils.config import ANN_DIR, IMG_DIR, TRAIN_DIR, VAL_DIR
from mmseg_utils.dataset_creation.dataset_utils import process_dataset_images


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("config1", type=Path)
    parser.add_argument("config2", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--vis-stride", type=int, default=10)

    args = parser.parse_args()
    return args


def extract_value(lines, value):
    for line in lines:
        if value in line:
            splits = line.split("=")
            if len(splits) != 2:
                continue
            splits = [x.strip() for x in splits]

            if splits[0] != value:
                continue

            data_root = splits[1]
            return data_root


def merge_folders(
    input_folders: typing.List[Path], output_folder: Path, num_zeros: int = 6
):
    # Collect all the files and sort them
    # count the number of leading numbers
    # Replace that with assending values
    # link to original data
    files = [input_folder.glob("*") for input_folder in input_folders]
    files = sorted(itertools.chain.from_iterable(files))

    output_folder.mkdir(exist_ok=True, parents=True)
    for i, file in enumerate(files):
        input_filename = file.name
        output_filename = str(i).zfill(num_zeros) + input_filename[num_zeros:]
        output_filepath = Path(output_folder, output_filename)
        os.symlink(file, output_filepath)


def main(config1: Path, config2: Path, output_folder: Path, vis_stride: int = 1):
    with open(config1, "r") as c1:
        cf1_lines = c1.readlines()
    with open(config2, "r") as c2:
        cf2_lines = c2.readlines()

    cf1_root = Path(extract_value(cf1_lines, "data_root")[1:-1])
    cf2_root = Path(extract_value(cf2_lines, "data_root")[1:-1])

    cf1_classes = eval(extract_value(cf1_lines, "classes"))
    cf2_classes = eval(extract_value(cf2_lines, "classes"))
    if cf1_classes != cf2_classes:
        raise NotImplementedError("Different classes is not yet supported")

    if output_folder.is_dir():
        shutil.rmtree(output_folder)

    for sub_folder in (
        (IMG_DIR, TRAIN_DIR),
        (IMG_DIR, VAL_DIR),
        (ANN_DIR, TRAIN_DIR),
        (ANN_DIR, VAL_DIR),
    ):
        merge_folders(
            (Path(cf1_root, *sub_folder), Path(cf2_root, *sub_folder)),
            Path(output_folder, *sub_folder),
        )

    process_dataset_images(
        Path(output_folder, IMG_DIR, TRAIN_DIR),
        class_names=cf1_classes,
        output_folder=output_folder,
        vis_stride=vis_stride,
    )


if __name__ == "__main__":
    args = parse_args()

    main(**args.__dict__)
