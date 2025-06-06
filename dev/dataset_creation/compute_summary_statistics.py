import argparse

from segmentation_utils.dataset_creation.summary_statistics import (
    compute_summary_statistics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir")
    parser.add_argument("--num-files", default=500, type=int)
    parser.add_argument("--extension", default="png")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    means, stds = compute_summary_statistics(
        args.images_dir, args.num_files, extension=args.extension
    )
    print(f"means: {means}\n, stds: {stds}")
