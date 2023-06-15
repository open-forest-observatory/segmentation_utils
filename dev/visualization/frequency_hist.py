from mmseg_utils.visualization.visualize_classes import show_frequency_hist
from mmseg_utils.config import CLASS_NAMES, PALETTE_MAP
import numpy as np

dataset = "safeforest23"
class_names = CLASS_NAMES[dataset]
palette = PALETTE_MAP[dataset]

freq = np.array(
    [
        0.25569415,
        0.02585316,
        0.00557238,
        0.02297996,
        0.32792042,
        0.0,
        0.00085417,
        0.00060204,
        0.0122692,
        0.32871919,
        0.0,
        0.0,
        0.0,
        0.01953534,
        0.0,
    ]
)
show_frequency_hist(palette=palette, class_names=class_names, freqs=freq)
