import numpy as np

from segmentation_utils.config import CLASS_NAMES, PALETTE_MAP
from segmentation_utils.visualization.visualize_classes import show_colormaps_flat

# weights = np.array(
#    [
#        0.12210163,
#        0.01234566,
#        0.00266098,
#        0.01097362,
#        0.15659184,
#        0.0,
#        0.00040789,
#        0.00028749,
#        0.00585891,
#        0.15697328,
#        0.0,
#        0.0,
#        0.0,
#        0.00932871,
#        0.0,
#    ]
# )
# print(weights / np.sum(weights))
dataset = "safeforest23_condensed"
class_names = CLASS_NAMES[dataset]
palette = PALETTE_MAP[dataset]
# mask = np.array([True, True, True, False])
mask = np.array(
    [
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
)
mask = np.array([True, True, True, False])
# mask = None

show_colormaps_flat(seg_map=palette, class_names=class_names, mask=mask)
