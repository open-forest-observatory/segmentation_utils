import argparse
import shutil
from pathlib import Path

import numpy as np

from segmentation_utils.dataset_creation.class_utils import remap_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder")
    parser.add_argument("--output-folder")
    parser.add_argument("--remap", nargs="+", type=int)
    parser.add_argument("--output-image-size", nargs="+", type=int)
    parser.add_argument(
        "--remap-mmseg-folder",
        action="store_true",
        help="Remap a train/test split folder structure",
    )
    parser.add_argument("--glob-str", default="*")

    args = parser.parse_args()
    return args


def main(
    input_folder,
    output_folder,
    remap,
    remap_mmseg_folder=False,
    output_image_size=None,
    glob_str="*",
):
    if remap is not None:
        remap = np.array(remap)
    if remap_mmseg_folder:
        input_ann_train = Path(input_folder, "ann_dir", "train")
        input_ann_val = Path(input_folder, "ann_dir", "val")

        input_img_train = Path(input_folder, "img_dir", "train")
        input_img_val = Path(input_folder, "img_dir", "val")

        output_ann_train = Path(output_folder, "ann_dir", "train")
        output_ann_val = Path(output_folder, "ann_dir", "val")

        output_img_train = Path(output_folder, "img_dir", "train")
        output_img_val = Path(output_folder, "img_dir", "val")

        remap_folder(
            input_ann_train,
            output_ann_train,
            remap,
            output_img_size=output_image_size,
            glob_str=glob_str,
        )
        remap_folder(
            input_ann_val,
            output_ann_val,
            remap,
            output_img_size=output_image_size,
            glob_str=glob_str,
        )

        shutil.copytree(input_img_train, output_img_train)
        shutil.copytree(input_img_val, output_img_val)
    else:
        remap_folder(
            input_folder,
            output_folder,
            remap,
            output_img_size=output_image_size,
            glob_str=glob_str,
        )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.input_folder,
        args.output_folder,
        args.remap,
        remap_mmseg_folder=args.remap_mmseg_folder,
        output_image_size=args.output_image_size,
        glob_str=args.glob_str,
    )
