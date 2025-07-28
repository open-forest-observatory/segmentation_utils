import argparse

from segmentation_utils.visualization.visualize_classes import show_colormaps

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmap-name")
    parser.add_argument("--class-names", nargs="+")
    parser.add_argument("--savepath")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    show_colormaps(**args.__dict__)