import numpy as np

RGB_EXT = "_rgb"
SEG_EXT = "_segmentation"
IMG_DIR = "img_dir"
ANN_DIR = "ann_dir"
TRAIN_DIR = "train"
VAL_DIR = "val"

# TODO update this to the real VIAME labels
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
    ]  # Obstacles
)

SAFEFOREST_23_CONDENSED_PALETTE = np.array(
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
    ]  # Obstacles
)
PALETTE_MAP = {"safeforest_23": SAFEFOREST_23_PALETTE}

IGNORE_INDEX = 255

BASIC_CLASS_MAP = {
    "Fuel": 0,
    "Canopy": 1,
    "Background": 2,
    "Trunks": 3,
    "unknown": 2,
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


ALL_CLASS_MAPS = {
    "basic": BASIC_CLASS_MAP,
    "safeforest23": SAFEFOREST_23_CLASS_MAP,
    "safeforest23_condensed": SAFEFOREST_CONDENSED_23_CLASS_MAP,
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
