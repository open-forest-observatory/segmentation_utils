"""
Show the segmentation labels from TartanAir to see if they have 
Any semantic meaning
"""
import argparse
from pathlib import Path

from mmseg_utils.config import PALETTE_MAP
from mmseg_utils.visualization.visualize_classes import visualize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seg-dir", type=Path, required=True)
    parser.add_argument(
        "--palette", default="matplotlib", choices=PALETTE_MAP.keys(), type=str
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
