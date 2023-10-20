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

freq_test = np.array(
    [
        2.73017106e-01,
        2.33781544e-03,
        3.06607209e-04,
        2.96282623e-02,
        3.86683891e-01,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        4.00330552e-05,
        1.84171402e-01,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
    ]
)
show_frequency_hist(
    palette=palette,
    class_names=class_names,
    freqs=freq_test,
    savefig="vis/test_freq.png",
)
