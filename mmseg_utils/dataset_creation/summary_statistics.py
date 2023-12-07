from pathlib import Path

import numpy as np
from imageio import imread
from numpy.random import choice
from tqdm import tqdm


def compute_summary_statistics(images, num_files=50, savepath=None, extension=""):
    files = [x for x in Path(images).glob("*" + extension) if x.is_file()]
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
