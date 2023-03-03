import numpy as np

RGB_EXT = "_rgb"
SEG_EXT = "_segmentation"
IMG_DIR = "img_dir"
ANN_DIR = "ann_dir"
TRAIN_DIR = "train"
VAL_DIR = "val"

# TODO update this to the real VIAME labels
SAFEFOREST_23_PALETTE = (np.random.random((20, 3)) * 255).astype(np.uint8)

PALETTE_MAP = {"safeforest_23": SAFEFOREST_23_PALETTE}
