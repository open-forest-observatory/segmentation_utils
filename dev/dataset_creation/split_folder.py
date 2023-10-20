import argparse
from ast import AsyncFunctionDef
from codecs import ignore_errors
from ctypes.wintypes import RGB
from genericpath import exists
from pathlib import Path
from mmseg_utils.config import RGB_EXT, SEG_EXT, IMG_DIR, ANN_DIR, TRAIN_DIR, VAL_DIR
from tqdm import tqdm
from imageio import imread
import numpy as np
import shutil
import os

from mmseg_utils.visualization.visualize_classes import load_png_npy

# IMAGE_FOLDER = Path(
#    "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/gascola/images_saved"
# )
# LABELS_FOLDER = Path(
#    "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/gascola/renders"
# )
# OUTPUT_FOLDER = Path(
#    "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/gascola/training"
# )
# TRAIN_FRAC = 0.8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-folder")
    parser.add_argument("--labels-folder")
    parser.add_argument("--output-folder")
    parser.add_argument("-train-frac", type=float, default=0.8)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    image_files = np.array(sorted(Path(args.images_folder).glob("*")))
    label_files = np.array(sorted(Path(args.labels_folder).glob("*")))

    valid_labels = np.array(
        list(
            map(
                lambda x: np.any(load_png_npy(x) != 255),
                tqdm(label_files),
            )
        )
    )

    image_files = image_files[valid_labels]
    label_files = label_files[valid_labels]
    n_valid = len(image_files)
    training_images = np.random.permutation(n_valid) < (n_valid * args.train_frac)

    IMG_TRAIN = Path(args.output_folder, IMG_DIR, TRAIN_DIR)
    IMG_VAL = Path(args.output_folder, IMG_DIR, VAL_DIR)
    ANN_TRAIN = Path(args.output_folder, ANN_DIR, TRAIN_DIR)
    ANN_VAL = Path(args.output_folder, ANN_DIR, VAL_DIR)
    [
        shutil.rmtree(x, ignore_errors=True)
        for x in (IMG_TRAIN, IMG_VAL, ANN_TRAIN, ANN_VAL)
    ]
    [os.makedirs(x, exist_ok=True) for x in (IMG_TRAIN, IMG_VAL, ANN_TRAIN, ANN_VAL)]

    for i in range(n_valid):
        filestem = f"{image_files[i].name}{RGB_EXT}{image_files[i].suffix}"
        os.symlink(
            image_files[i],
            Path(
                (IMG_TRAIN if training_images[i] else IMG_VAL),
                f"{image_files[i].stem}{RGB_EXT}{image_files[i].suffix}",
            ),
        )
        os.symlink(
            label_files[i],
            Path(
                (ANN_TRAIN if training_images[i] else ANN_VAL),
                f"{image_files[i].stem}{SEG_EXT}{image_files[i].suffix}",
            ),
        )
