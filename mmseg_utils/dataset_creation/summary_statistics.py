from imageio import imread
import numpy as np
from tqdm import tqdm
from pathlib import Path
from numpy.random import choice


def compute_summary_statistics(images, num_files=500, savepath=None, extension="png"):
    files = list(Path(images).glob("*." + extension))
    files = choice(files, min(num_files, len(files)))

    imgs = [imread(x) for x in tqdm(files)]
    imgs = np.concatenate(imgs, axis=0)  # tile vertically
    mean = np.mean(imgs, axis=(0, 1))
    std = np.std(imgs, axis=(0, 1))

    means, stds = mean.tolist(), std.tolist()
    if savepath is not None:
        with open(savepath, "w") as outfile_h:
            outfile_h.write(f"mean per channel: {means}\n")
            outfile_h.write(f"std per channel: {stds}\n")
    return means, stds
