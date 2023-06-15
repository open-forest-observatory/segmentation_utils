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
    output = np.ones((index_image.shape[0], 3)) * 255
    colored_image = palette[index_image[dont_ignore]]
    output[dont_ignore] = colored_image
    colored_image = np.reshape(output, (h, w, 3))
    return colored_image.astype(np.uint8)


def blend_images(im1, im2, alpha=0.7):
    return (alpha * im1 + (1 - alpha) * im2).astype(np.uint8)


def show_frequency_hist(palette, class_names, freqs, savefig=None):
    mask = freqs != 0
    palette = palette[mask]
    class_names = np.array(class_names)[mask]
    freqs = freqs[mask]

    order = np.array(list(reversed(np.argsort(freqs).tolist())))
    palette = palette[order]
    class_names = class_names[order].tolist()
    freqs = freqs[order]

    plt.bar(
        np.arange(palette.shape[0]),
        height=freqs,
        color=palette / 255.0,
        tick_label=class_names,
    )
    plt.xticks(rotation=45, size=14)  # Rotates X-Axis Ticks by 45-degrees

    plt.ylabel("Class fraction", size=14)
    plt.yticks(size=14)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()


def show_colormaps_flat(seg_map, class_names, mask=None, savepath=None):
    if mask is not None:
        seg_map = seg_map[mask]
        class_names = np.array(class_names)[mask]

    num_classes = len(class_names)
    fig, axs = plt.subplots(1, num_classes)

    for index in range(num_classes):

        color = seg_map[index]
        color = np.expand_dims(color, (0, 1))
        vis_square = np.repeat(
            np.repeat(color, repeats=100, axis=0), repeats=100, axis=1
        )
        axs[index].imshow(vis_square)
        axs[index].set_title(class_names[index])
        axs[index].axis("off")

    plt.axis("off")
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)
        plt.close()


def show_colormaps(seg_map, class_names, savepath=None):
    num_classes = len(class_names)
    n_squares = int(np.ceil(np.sqrt(num_classes)))
    fig, axs = plt.subplots(n_squares, n_squares)

    for index in range(num_classes):
        i = index // n_squares
        j = index % n_squares

        color = seg_map[index]
        color = np.expand_dims(color, (0, 1))
        vis_square = np.repeat(
            np.repeat(color, repeats=100, axis=0), repeats=100, axis=1
        )
        axs[i, j].imshow(vis_square)
        axs[i, j].set_title(class_names[index])
        axs[i, j].axis("off")
    # Clear remaining subplots
    for index in range(num_classes, n_squares * n_squares):
        i = index // n_squares
        j = index % n_squares
        axs[i, j].axis("off")

    plt.axis("off")
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)
        plt.close()


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
        img = np.flip(imread(image_file), axis=2)

        vis_seg = visualize_with_palette(seg, palette)
        blended = blend_images_gray(img, vis_seg, alpha)

        concat = np.concatenate((img, vis_seg, blended), axis=0)
        savepath = output_dir.joinpath(image_file.name)
        gt_classes_savepath = output_dir.joinpath(
            image_file.name.replace(".png", "_vis_seg.png")
        )
        imwrite(str(savepath), concat)
        imwrite(str(gt_classes_savepath), vis_seg)


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
