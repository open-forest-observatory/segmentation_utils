import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

from mmseg_utils.config import PALETTE_MAP
from mmseg_utils.dataset_creation.file_utils import ensure_dir_normal_bits
from mmseg_utils.utils.files import get_matching_files


def visualize_with_cmap(index_image, cmap_name, ignore_ind=255):
    """
    index_image : np.ndarray
        The predicted semantic map with indices. (H,W)
    cmap_name : np.ndarray
        The colors for each index. (N classes,3)
    """
    # Get the colormapping object
    cmap = plt.get_cmap(cmap_name)
    # Actually do the colormapping
    # TODO see if this will out return out of range error for the null index
    # Extract the first three channels, dropping alpha
    colored_image = cmap(index_image, bytes=True)[..., :3]

    # Compute which enties are not the ignore ind
    dont_ignore = index_image != ignore_ind

    # No ignore inds entries
    if np.all(dont_ignore):
        output = colored_image
    else:
        # Create an output array of zeros
        output = np.zeros(
            (index_image.shape[0], index_image.shape[1], 3), dtype=np.uint8
        )
        # Set the colored values for the non-ignored indices
        output[dont_ignore] = colored_image[dont_ignore]
    return output


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


def show_colormaps(cmap_name, class_names, savepath=None):
    num_classes = len(class_names)
    n_squares = int(np.ceil(np.sqrt(num_classes)))
    matplotlib.use("Agg")
    _, axs = plt.subplots(n_squares, n_squares)

    cmap = plt.get_cmap(cmap_name)
    for index in range(num_classes):
        i = index // n_squares
        j = index % n_squares

        color = cmap(index)
        color = np.expand_dims(color, (0, 1))
        vis_square = np.repeat(
            np.repeat(color, repeats=100, axis=0), repeats=100, axis=1
        )
        axs[i, j].imshow(vis_square)
        axs[i, j].set_title(class_names[index], fontsize=8)
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
    elif filename.suffix in (".png", ".jpg", ".jpeg", ".JPG"):
        return np.array(Image.open(filename))


def visualize(
    seg_dir,
    image_dir,
    output_dir,
    cmap_name="tab10",
    alpha=0.5,
    vis_number=10,
    image_extension="",
    label_extension="",
    ignore_substr_images_for_matching="",
    ignore_substr_labels_for_matching="",
    remove_dir: bool = True,
):
    if remove_dir and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    ensure_dir_normal_bits(output_dir)
    image_files, seg_files = get_matching_files(
        images_folder=image_dir,
        labels_folder=seg_dir,
        image_extension=image_extension,
        label_extension=label_extension,
        ignore_substr_images=ignore_substr_images_for_matching,
        ignore_substr_labels=ignore_substr_labels_for_matching,
    )

    stride = int(max(1, len(seg_files) / vis_number))

    for seg_file, image_file in tqdm(
        list(zip(seg_files, image_files))[::stride],
        total=len(seg_files[::stride]),
        desc=f"visualizing to {output_dir}",
    ):
        seg = load_png_npy(seg_file)
        img = np.array(Image.open(image_file))

        vis_seg = visualize_with_cmap(seg, cmap_name)
        blended = blend_images_gray(img, vis_seg, alpha)

        concat = np.concatenate((vis_seg, img, blended), axis=1)
        savepath = Path(output_dir, Path(image_file).relative_to(image_dir))
        savepath.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(concat).save(str(savepath))


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
