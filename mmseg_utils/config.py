import numpy as np

RGB_EXT = "_rgb"
SEG_EXT = "_segmentation"
IMG_DIR = "img_dir"
ANN_DIR = "ann_dir"
TRAIN_DIR = "train"
VAL_DIR = "val"

IGNORE_INDEX = 255

SAFEFOREST_23_PALETTE = np.array(
    [
        [128, 224, 255],  # Dry Grass
        [0, 255, 255],  # Green Grass (canopy)
        [80, 0, 255],  # Dry Shrubs
        [45, 112, 134],  # Green Shrubs
        [0, 255, 144],  # Canopy
        [128, 255, 199],  # Wood Pieces
        [224, 0, 255],  # Litterfall (bare earth or fuel)
        [0, 194, 255],  # Timber Litter
        [45, 134, 95],  # Live Trunks
        [255, 0, 111],  # Bare Earth
        [239, 128, 255],  # People
        [167, 128, 255],  # Sky
        [134, 45, 83],  # Blurry
        [45, 68, 134],
    ]
)

SAFEFOREST_23_CONDENSED_PALETTE = np.array(
    [
        [255, 0, 0],  # Fuel
        [0, 255, 0],  # Canopy
        [0, 0, 0],  # Background
        [255, 0, 255],  # Trunks
    ]
)

BASIC_CLASS_MAP = {
    "Fuel": 0,
    "Canopy": 1,
    "Background": 2,
    "Trunks": 3,
    "unknown": IGNORE_INDEX,
}

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
    "Sky": 11,
    "Blurry": 12,
    "Obstacle": 13,
    "Obstacles": 13,
    "Drone": 14,
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
    "Sky": 2,
    "Blurry": 2,
    "Obstacle": 2,
    "Obstacles": 2,
    "Drone": 2,
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
PALETTE_MAP = {
    "basic": SAFEFOREST_23_CONDENSED_PALETTE,
    "safeforest23": SAFEFOREST_23_PALETTE,
    "safeforest23_condensed": SAFEFOREST_23_CONDENSED_PALETTE,
}
