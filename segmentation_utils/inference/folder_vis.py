from argparse import ArgumentParser
from pathlib import Path

from mmseg.apis import inference_model, init_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="/ofo-share/repos-david/Safeforest_CMU_data_dvc/models/segformer_mit-b5_8xb1-20k_safeforest23-1024x1024/segformer_mit-b5_8xb1-20k_safeforest23-1024x1024.py",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="/ofo-share/repos-david/Safeforest_CMU_data_dvc/models/segformer_mit-b5_8xb1-20k_safeforest23-1024x1024/iter_20000.pth",
    )
    parser.add_argument(
        "--image-folder",
        default="/ofo-share/repos-david/Safeforest_CMU_data_dvc/data/site_Gascola/04_27_23/collect_03/processed_01/images/left_camera_0.5",
    )
    args = parser.parse_args()
    return args


def main(
    config_path,
    checkpoint_path,
    image_folder=None,
):
    files = Path(image_folder).glob("*")
    # init model and load checkpoint
    model = init_model(config_path, checkpoint_path)
    breakpoint()
    for img_path in files:
        result = inference_model(model, img_path)


if __name__ == "__main__":
    args = parse_args()
    main(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        image_folder=args.image_folder,
    )
