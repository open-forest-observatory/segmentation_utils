import numpy as np
import time
from mmseg_utils.dataset_creation.file_utils import ensure_dir_normal_bits
from imageio import imwrite, imread
from glob import glob
import time
import matplotlib.pyplot as plt
from mmseg_utils.config import PALETTE_MAP
from pathlib import Path
from tqdm import tqdm


def visualize_with_palette(index_image, palette, ignore_ind=255):
    """
    index_image : np.ndarray
        The predicted semantic map with indices. (H,W)
    palette : np.ndarray
        The colors for each index. (N classes,3)
    """
    h, w = index_image.shape
    index_image = index_image.flatten()

    dont_ignore = index_image != ignore_ind
    output = np.zeros((index_image.shape[0], 3))
    colored_image = palette[index_image[dont_ignore]]
    output[dont_ignore] = colored_image
    colored_image = np.reshape(output, (h, w, 3))
    return colored_image.astype(np.uint8)


def blend_images(im1, im2, alpha=0.7):
    return (alpha * im1 + (1 - alpha) * im2).astype(np.uint8)


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


def blend_images_gray(im1, im2, alpha=0.7):
    """Blend two images with the first transformed to grayscale

    im1: img to be turned to gray
    im2: img kept as normal color
    alpha: contribution of first image
    """
    num_channels = im1.shape[2]
    im1 = np.mean(im1, axis=2)
    im1 = np.expand_dims(im1, axis=2)
    im1 = np.repeat(im1, repeats=num_channels, axis=2)
    return (alpha * im1 + (1 - alpha) * im2).astype(np.uint8)
