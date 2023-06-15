import argparse
from pathlib import Path
from mmseg_utils.dataset_creation.class_utils import remap_classes_bool_indexing
import shutil
import os
from imageio import imwrite, imread
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder")
    parser.add_argument("--output-folder")
    parser.add_argument("--remap", nargs="+", type=int)
    args = parser.parse_args()
    return args

def remap_folder(input_folder, output_folder, remap, glob="*"):
    os.makedirs(output_folder, exist_ok=True)
    input_files = Path(input_folder).glob(glob)
    for input_file in input_files:
        input_label = imread(input_file)
        output_label = remap_classes_bool_indexing(input_label, remap)
        output_file = Path(output_folder, input_file.name)
        imwrite(output_file, output_label)

def main(input_folder, output_folder, remap):
    remap = np.array(remap)
    input_ann_train = Path(input_folder, "ann_dir", "train")
    input_ann_val = Path(input_folder, "ann_dir", "val")

    input_img_train = Path(input_folder, "img_dir", "train")
    input_img_val = Path(input_folder, "img_dir", "val")

    output_ann_train = Path(output_folder, "ann_dir", "train")
    output_ann_val = Path(output_folder, "ann_dir", "val")

    output_img_train = Path(output_folder, "img_dir", "train")
    output_img_val = Path(output_folder, "img_dir", "val")

    remap_folder(input_ann_train, output_ann_train, remap)
    remap_folder(input_ann_val, output_ann_val, remap)

    shutil.copytree(input_img_train, output_img_train)
    shutil.copytree(input_img_val, output_img_val)

if __name__ == "__main__":
    args = parse_args()
    main(args.input_folder, args.output_folder, args.remap)
