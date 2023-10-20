import argparse
from pathlib import Path
import shutil
import numpy as np
from mmseg_utils.dataset_creation.class_utils import remap_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder")
    parser.add_argument("--output-folder")
    parser.add_argument("--remap", nargs="+", type=int)
    parser.add_argument("--output-image-size", nargs="+", type=int)
    parser.add_argument("--remap-tree", action="store_true")

    args = parser.parse_args()
    return args


def main(input_folder, output_folder, remap, remap_tree=False, output_image_size=None):
    if remap is not None:
        remap = np.array(remap)
    if remap_tree:
        input_ann_train = Path(input_folder, "ann_dir", "train")
        input_ann_val = Path(input_folder, "ann_dir", "val")

        input_img_train = Path(input_folder, "img_dir", "train")
        input_img_val = Path(input_folder, "img_dir", "val")

        output_ann_train = Path(output_folder, "ann_dir", "train")
        output_ann_val = Path(output_folder, "ann_dir", "val")

        output_img_train = Path(output_folder, "img_dir", "train")
        output_img_val = Path(output_folder, "img_dir", "val")

        remap_folder(
            input_ann_train, output_ann_train, remap, output_img_size=output_image_size
        )
        remap_folder(
            input_ann_val, output_ann_val, remap, output_img_size=output_image_size
        )

        shutil.copytree(input_img_train, output_img_train)
        shutil.copytree(input_img_val, output_img_val)
    else:
        remap_folder(
            input_folder, output_folder, remap, output_img_size=output_image_size
        )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.input_folder,
        args.output_folder,
        args.remap,
        remap_tree=args.remap_tree,
        output_image_size=args.output_image_size,
    )
