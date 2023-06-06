import argparse
from imageio import imread
import numpy as np
from tqdm import tqdm
from pathlib import Path
from numpy.random import choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--num-files", default=500, type=int)
    args = parser.parse_args()
    return args


def main(images, num_files, extension="*.png"):
    files = list(Path(images).glob(extension))

    files = choice(files, min(num_files, len(files)))

    imgs = [imread(x) for x in tqdm(files)]
    imgs = np.concatenate(imgs, axis=0)  # tile vertically
    mean = np.mean(imgs, axis=(0, 1))
    std = np.std(imgs, axis=(0, 1))
    print(f"mean: {mean.tolist()}, stdev: {std.tolist()}")


if __name__ == "__main__":
    args = parse_args()
    main(args.images_dir, args.num_files)
