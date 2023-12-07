import numpy as np

from mmseg_utils.dataset_creation.class_utils import remap_folder

INPUT_FOLDER = "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/gascola/renders"
OUTPUT_FOLDER = (
    "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/gascola/renders_remapped"
)
REMAP = np.arange(10)

remap_folder(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    remap=REMAP,
)
