from mmseg_utils.dataset_creation.class_utils import remap_folder
import numpy as np

INPUT_FOLDER = "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/gascola/renders"
OUTPUT_FOLDER = (
    "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/gascola/renders_remapped"
)
REMAP = np.arange(3)

remap_folder(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    remap=REMAP,
    output_img_size=(2048, 2448),
)
