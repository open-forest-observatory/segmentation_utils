"""
Show the segmentation labels from TartanAir to see if they have 
Any semantic meaning
"""
import argparse
import time
from glob import glob
from pathlib import Path
from imageio import imwrite

import matplotlib.pyplot as plt
import numpy as np
from mmseg_utils.config import PALETTE_MAP
from mmseg_utils.dataset_creation.file_utils import ensure_dir_normal_bits
from mmseg_utils.visualization.visualize_classes import (
    blend_images_gray,
    visualize_with_palette,
)

from imageio import imread
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seg-dir", type=Path, required=True)
    parser.add_argument(
        "--palette", default="safeforest_23", choices=PALETTE_MAP.keys(), type=str
    )
    parser.add_argument(
        "--stride", default=1, type=int, help="Evaluate images every <stride>th image"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Contribution of first image to blended one",
    )
    args = parser.parse_args()
    return args


def show_colormaps(seg_map, num_classes=7):
    square_size = int(np.ceil(np.sqrt(num_classes)))
    vis = np.zeros((square_size, square_size, 3))
    for index in range(num_classes):
        i = index // square_size
        j = index % square_size
        vis[i, j] = seg_map[index]
    vis = vis.astype(np.uint8)
    vis = np.repeat(np.repeat(vis, repeats=100, axis=0), repeats=100, axis=1)
    plt.imshow(vis)
    plt.show()
    breakpoint()


def load_png_npy(filename):
    if filename.suffix == ".npy":
        return np.load(filename)
    elif filename.suffix in (".png", ".jpg", ".jpeg"):
        return imread(filename)


def visualize(seg_dir, image_dir, output_dir, palette_name="rui", alpha=0.5, stride=1):
    palette = PALETTE_MAP[palette_name]
    ensure_dir_normal_bits(output_dir)
    seg_files = sorted(
        list(Path(seg_dir).glob("*.npy")) + list(Path(seg_dir).glob("*.png"))
    )
    image_files = sorted(Path(image_dir).glob("*.png"))
    if len(seg_files) != len(image_files):
        raise ValueError(
            f"Different length inputs, {len(seg_files)}, {len(image_files)}"
        )

    for seg_file, image_file in tqdm(
        list(zip(seg_files, image_files))[::stride], total=len(seg_files[::stride])
    ):
        seg = load_png_npy(seg_file)
        img = imread(image_file)

        vis_seg = visualize_with_palette(seg, palette)
        blended = blend_images_gray(img, vis_seg, alpha)

        concat = np.concatenate((img, vis_seg, blended), axis=0)
        savepath = output_dir.joinpath(image_file.name)
        imwrite(str(savepath), np.flip(concat, axis=2))


if __name__ == "__main__":
    args = parse_args()
    visualize(
        args.seg_dir,
        args.image_dir,
        args.output_dir,
        args.palette,
        args.alpha,
        args.stride,
    )
