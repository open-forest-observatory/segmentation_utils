import argparse
from segmentation_utils.dataset_creation.folder_to_cityscapes import folder_to_cityscapes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-folder")
    parser.add_argument("--labels-folder")
    parser.add_argument("--output-folder")
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--image-ext", nargs="+", default="JPG")
    parser.add_argument("--label-ext", nargs="+", default="png")
    parser.add_argument("--remove-old", action="store_true")
    parser.add_argument("--train-frac", type=float, default=1.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--vis-number", type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    folder_to_cityscapes(
        args.images_folder,
        args.labels_folder,
        args.image_ext,
        args.label_ext,
        args.train_frac,
        args.val_frac,
        args.output_folder,
        args.remove_old,
        args.classes,
        args.vis_number,
    )
