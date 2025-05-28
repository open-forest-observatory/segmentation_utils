"""
Show the segmentation labels from TartanAir to see if they have
Any semantic meaning
"""

import argparse
from pathlib import Path

from segmentation_utils.config import PALETTE_MAP
from segmentation_utils.visualization.visualize_classes import visualize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seg-dir", type=Path, required=True)
    parser.add_argument("--palette", default="tab10", type=str)
    parser.add_argument(
        "--vis-number",
        default=10,
        type=int,
        help="Number of images you want to visualize",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Contribution of first image to blended one",
    )
    parser.add_argument("--ignore-substr-images-for-matching", default="", type=str)
    parser.add_argument("--ignore-substr-labels-for-matching", default="", type=str)
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
        args.vis_number,
        ignore_substr_images_for_matching=args.ignore_substr_images_for_matching,
        ignore_substr_labels_for_matching=args.ignore_substr_labels_for_matching,
    )
