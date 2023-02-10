import pandas as pd
from pathlib import Path
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

CSV_FILE = "/home/frc-ag-1/Downloads/oporto_2021_12_17_collect_1 (1).csv"
IMAGE_FOLDER = "/media/frc-ag-1/Elements/data/Safeforest_CMU_data_dvc/data/site_Oporto_clearing/2021_12_17/collect_1/processed_1/images/mapping_left"
# FILE = "/home/frc-ag-1/Downloads/oporto_2021_12_17_collect_1.dive.json"
# with open(FILE, "r") as f:
#    data = json.load(f)
data = pd.read_csv(
    CSV_FILE,
    sep=",",
    names=(
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
    ),
)
print(data)
breakpoint()
#data = pd.read_csv(
#    CSV_FILE,
#    sep=",",
#    skiprows=0,
#    names=(
#        "track_ID",
#        "image",
#        "frame_ID",
#        "bbox_tl_x",
#        "bbox_tl_y",
#        "bbox_br_x",
#        "bbox_br_y",
#        "class",
#        "confidence",
#        "polygon",
#    ),
#)
print(data)
for name, row in data.iterrows():
    polygon = np.array(row["polygon"][7:].split(" ")).astype(int)
    polygon = np.reshape(polygon, (int(polygon.shape[0] / 2), 2))
    img_file = row["image"]
    breakpoint()
    image_path = Path(IMAGE_FOLDER, img_file)
    img = imread(image_path)
    plt.imshow(img)

breakpoint()
