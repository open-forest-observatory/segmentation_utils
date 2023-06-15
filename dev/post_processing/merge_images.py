from pathlib import Path
from imageio import imread, imwrite
import numpy as np
import os


# IMG_PATH = "/home/frc-ag-1/data/Safeforest_CMU_data_dvc/models/segnext_mscan-t_1xb16-adamw-160k_safeforest_gascola_23_04_27_collect_04-512x512/vis_collect_05_gt"
# IMG_GLOB = "*_rgb/condensed_*_rgb_img"
#
# FULL_CLASS_PREDS_PATH = "/home/frc-ag-1/data/Safeforest_CMU_data_dvc/models/segnext_mscan-t_1xb16-adamw-160k_safeforest_gascola_23_04_27_collect_04-512x512/vis_collect_05_gt"
# FULL_CLASS_PREDS_GLOB = "*_rgb/*img.png"
#
# COMPRESSED_PREDS_PATH = "/home/frc-ag-1/data/Safeforest_CMU_data_dvc/models/segnext_mscan-t_1xb16-adamw-160k_safeforest_gascola_23_04_27_collect_04-512x512/vis_collect_05_gt"
# COMPRESSED_PREDS_GLOB = "*_rgb"

FULL_CLASS_GT_PATH = Path(
    "/home/frc-ag-1/data/Safeforest_CMU_data_dvc/models/segnext_mscan-t_1xb16-adamw-160k_safeforest_gascola_23_04_27_collect_04-512x512/vis_collect_05_gt/gt_seg_vis"
)
COMPRESSED_GT_PATH = Path(
    "/home/frc-ag-1/data/Safeforest_CMU_data_dvc/models/segnext_mscan-t_1xb16-adamw-160k_safeforest_gascola_23_04_27_collect_04-512x512/vis_collect_05_gt/gt_seg_vis_condensed"
)

FOLDER_PATH = "/home/frc-ag-1/data/Safeforest_CMU_data_dvc/models/segnext_mscan-t_1xb16-adamw-160k_safeforest_gascola_23_04_27_collect_04-512x512/vis_collect_05_gt/preds_brightened_1.000000"
OUTPUT_FOLDER = "/home/frc-ag-1/data/Safeforest_CMU_data_dvc/models/segnext_mscan-t_1xb16-adamw-160k_safeforest_gascola_23_04_27_collect_04-512x512/vis_collect_05_gt/vis_concatenated"

files = list(Path(FOLDER_PATH).rglob("*png"))

img_files = [f for f in files if ("rgb_img" in f.name and "condensed" not in f.name)]
compressed_pred_files = [
    f for f in files if ("condensed" in f.name and "img" not in f.name)
]
full_class_pred_files = [
    f for f in files if ("condensed" not in f.name and "img" not in f.name)
]
full_class_gts = list(FULL_CLASS_GT_PATH.glob("*vis_seg.png"))
compressed_gts = list(COMPRESSED_GT_PATH.glob("*vis_seg.png"))

img_files = sorted(img_files)
compressed_pred_files = sorted(compressed_pred_files)
full_class_pred_files = sorted(full_class_pred_files)
compressed_gts = sorted(compressed_gts)
full_class_gts = sorted(full_class_gts)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for i in range(len(img_files)):
    rgb = imread(img_files[i])
    cp = imread(compressed_pred_files[i])
    fcp = imread(full_class_pred_files[i])
    cg = imread(compressed_gts[i])
    fcg = imread(full_class_gts[i])
    spacer = np.ones((rgb.shape[0], 50, 3), dtype=np.uint8) * 255

    concatenated = np.concatenate(
        (rgb, spacer, fcg, spacer, fcp, spacer, cg, spacer, cp), axis=1
    )
    output_file = Path(OUTPUT_FOLDER, img_files[i].name)
    imwrite(output_file, concatenated)
