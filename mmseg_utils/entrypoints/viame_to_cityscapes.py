from argparse import ArgumentParser

from mmseg_utils.dataset_creation.parsing import parse_viame_annotations_dataset
from mmseg_utils.dataset_creation.dataset_utils import process_dataset_images


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--image-extension", default="jpg")
    parser.add_argument("--class-map")
    parser.add_argument(
        "--encode-rotation-with-metadata",
        action="store_true",
        help="Should the rotation of the label be encoded in the image metadata rather than by rotating the pixels",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    parse_viame_annotations_dataset(**args.__dict__)
