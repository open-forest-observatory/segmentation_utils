import argparse
from ast import AsyncFunctionDef
from codecs import ignore_errors
from ctypes.wintypes import RGB
from genericpath import exists
from pathlib import Path
from mmseg_utils.dataset_creation.mmseg_config import create_new_config
from mmseg_utils.dataset_creation.summary_statistics import compute_summary_statistics
from mmseg_utils.config import (
    MATPLOTLIB_PALLETE,
    RGB_EXT,
    SEG_EXT,
    IMG_DIR,
    ANN_DIR,
    TRAIN_DIR,
    VAL_DIR,
)
from tqdm import tqdm
from imageio import imread
import numpy as np
import shutil
import os
from mmseg_utils.utils.files import get_matching_files

from mmseg_utils.visualization.visualize_classes import (
    load_png_npy,
    show_colormaps,
    visualize,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-folder")
    parser.add_argument("--labels-folder")
    parser.add_argument("--output-folder")
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--image-ext", default="JPG")
    parser.add_argument("--label-ext", default="png")
    parser.add_argument("--remove-old", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--vis-stride", type=int, default=20)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    image_files, label_files = get_matching_files(
        args.images_folder, args.labels_folder, args.image_ext, args.label_ext
    )

    valid_labels = np.array(
        list(
            map(
                lambda x: np.any(load_png_npy(x) != 255),
                tqdm(label_files, desc="Checking for completely null images"),
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

    if args.remove_old:
        [
            shutil.rmtree(x, ignore_errors=True)
            for x in (IMG_TRAIN, IMG_VAL, ANN_TRAIN, ANN_VAL)
        ]
    [os.makedirs(x, exist_ok=True) for x in (IMG_TRAIN, IMG_VAL, ANN_TRAIN, ANN_VAL)]

    for i in tqdm(
        range(n_valid), desc="Linking files into either train or test folders"
    ):
        stem = image_files[i].relative_to(args.images_folder)
        stem = str(stem.with_suffix("")).replace(os.path.sep, "_")

        os.symlink(
            image_files[i],
            Path(
                (IMG_TRAIN if training_images[i] else IMG_VAL),
                f"{stem}{RGB_EXT}{image_files[i].suffix}",
            ),
        )
        os.symlink(
            label_files[i],
            Path(
                (ANN_TRAIN if training_images[i] else ANN_VAL),
                f"{stem}{SEG_EXT}{label_files[i].suffix}",
            ),
        )

    mean, std = compute_summary_statistics(images=IMG_TRAIN, num_files=20)
    print(f"mean: {mean}, std: {std}")
    if args.classes is not None:
        output_config = Path(args.output_folder, Path(args.output_folder).stem + ".py")
        print(f"About to save config to {output_config}")
        create_new_config(
            "configs/cityscapes_forests.py",
            output_config_file=output_config,
            mean=mean,
            std=std,
            classes=args.classes,
            data_root=args.output_folder,
        )
    vis_train = Path(args.output_folder, "vis", "train")
    vis_val = Path(args.output_folder, "vis", "val")
    vis_train.mkdir(exist_ok=True, parents=True)

    show_colormaps(
        MATPLOTLIB_PALLETE,
        class_names=args.classes,
        savepath=Path(args.output_folder, "colormap.png"),
    )
    visualize(
        ANN_TRAIN,
        IMG_TRAIN,
        vis_train,
        ignore_substr_images_for_matching=RGB_EXT,
        ignore_substr_labels_for_matching=SEG_EXT,
        stride=args.vis_stride,
    )
    visualize(
        ANN_VAL,
        IMG_VAL,
        vis_val,
        ignore_substr_images_for_matching=RGB_EXT,
        ignore_substr_labels_for_matching=SEG_EXT,
        stride=args.vis_stride,
    )
