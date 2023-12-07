import matplotlib
import matplotlib.pyplot as plt
import numpy as np

RGB_EXT = "_rgb"
SEG_EXT = "_segmentation"
IMG_DIR = "img_dir"
ANN_DIR = "ann_dir"
TRAIN_DIR = "train"
VAL_DIR = "val"

IGNORE_INDEX = 255

SAFEFOREST_23_PALETTE = np.flip(
    np.array(
        [
            [128, 224, 255],  # Dry Grass, 0
            [0, 255, 255],  # Green Grass (canopy), 1
            [80, 0, 255],  # Dry Shrubs, 2
            [45, 112, 134],  # Green Shrubs, 3
            [0, 255, 144],  # Canopy, 4
            [128, 255, 199],  # Wood Pieces, 5
            [224, 0, 255],  # Litterfall (bare earth or fuel), 6
            [0, 194, 255],  # Timber Litter, 7
            [45, 134, 95],  # Live Trunks, 8
            [255, 0, 111],  # Bare Earth, 9
            [239, 128, 255],  # People, 10
            [167, 128, 255],  # Sky, 11
            [134, 45, 83],  # Blurry, 12
            [83, 45, 134],  # Obstacle
            [45, 68, 134],  # Drones, 13
        ]
    ),
    axis=1,
)

SAFEFOREST_23_CONDENSED_PALETTE = np.array(
    [
        [255, 0, 0],  # Fuel
        [0, 255, 0],  # Canopy
        [0, 0, 0],  # Background
        [255, 0, 255],  # Trunks
    ]
)

SAFEFOREST_23_NAMES = [
    "Dry Grass",
    "Green Grass",
    "Dry Shrubs",
    "Green Shrubs",
    "Canopy",
    "Wood Pieces",
    "Litterfall",
    "Timber Litter",
    "Live Trunks",
    "Bare Earth",
    "People",
    "Sky",
    "Blurry",
    "Obstacle",
    "Drone",
]

SAFEFOREST_23_CONDENSED_NAMES = ["Fuel", "Canopy", "Background", "Trunks"]

BASIC_CLASS_MAP = {
    "Fuel": 0,
    "Canopy": 1,
    "Background": 2,
    "Trunks": 3,
    "unknown": IGNORE_INDEX,
}


def hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


MATPLOTLIB_PALLETE = (np.array(matplotlib.colormaps["tab20"].colors) * 255).astype(
    np.uint8
)


SAFEFOREST_23_CLASS_MAP = {
    "Dry Grass": 0,
    "Green Grass": 1,
    "Dry Shrubs": 2,
    "Green Shrubs": 3,
    "Canopy": 4,
    "Wood Pieces": 5,
    "Litterfall": 6,
    "Timber Litter": 7,
    "Live Trunks": 8,
    "Bare Earth": 9,
    "People": 10,
    "Human": 10,
    "Sky": 11,
    "Blurry": 12,
    "Obstacle": 13,
    "Obstacles": 13,
    "Drone": 14,
    "Fuel": IGNORE_INDEX,
    "unknown": IGNORE_INDEX,
}

SAFEFOREST_CONDENSED_23_CLASS_MAP = {
    "Dry Grass": 0,
    "Green Grass": 0,
    "Dry Shrubs": 0,
    "Green Shrubs": 1,
    "Canopy": 1,
    "Wood Pieces": 0,
    "Litterfall": 0,
    "Timber Litter": 0,
    "Live Trunks": 3,
    "Bare Earth": 2,
    "People": 2,
    "Human": 2,
    "Sky": 2,
    "Blurry": 2,
    "Obstacle": 2,
    "Obstacles": 2,
    "Drone": 2,
    "Fuel": 0,
    "unknown": IGNORE_INDEX,
}

COLUMN_NAMES = (
    "column_ID",
    "image_name",
    "frame_ID",
    "bbox_tl_x",
    "bbox_tl_y",
    "bbox_br_x",
    "bbox_br_y",
    "det_or_len_confidence",
    "length",
    "class",
    "class_confidence",
    "polygon",
)

CLASS_MAP = {
    "basic": BASIC_CLASS_MAP,
    "safeforest23": SAFEFOREST_23_CLASS_MAP,
    "safeforest23_condensed": SAFEFOREST_CONDENSED_23_CLASS_MAP,
}
CLASS_NAMES = {
    "basic": SAFEFOREST_23_CONDENSED_NAMES,
    "safeforest23": SAFEFOREST_23_NAMES,
    "safeforest23_condensed": SAFEFOREST_23_CONDENSED_NAMES,
}
PALETTE_MAP = {
    "basic": SAFEFOREST_23_CONDENSED_PALETTE,
    "matplotlib": MATPLOTLIB_PALLETE,
    "safeforest23": SAFEFOREST_23_PALETTE,
    "safeforest23_condensed": SAFEFOREST_23_CONDENSED_PALETTE,
}
