from pathlib import Path

from mmseg_utils.config import (
    ANN_DIR,
    IMG_DIR,
    RGB_EXT,
    SEG_EXT,
    TRAIN_DIR,
    VAL_DIR,
    DEFAULT_CITYSCAPES_CONFIG,
)
from mmseg_utils.dataset_creation.mmseg_config import create_new_config
from mmseg_utils.dataset_creation.summary_statistics import compute_summary_statistics
from mmseg_utils.visualization.visualize_classes import show_colormaps, visualize


def process_dataset_images(
    training_images_folder,
    class_names,
    output_folder,
    num_summary_files=20,
    summary_subsample=100,
    vis_stride=50,
    cmap_name="tab10",
):
    mean, std = compute_summary_statistics(
        images=training_images_folder,
        num_files=num_summary_files,
        subsample_step=summary_subsample,
    )
    print(f"mean: {mean}, std: {std}")

    if class_names is not None:
        output_config = Path(output_folder, Path(output_folder).stem + ".py")
        print(f"About to save config to {output_config}")
        create_new_config(
            DEFAULT_CITYSCAPES_CONFIG,
            output_config_file=output_config,
            mean=mean,
            std=std,
            classes=class_names,
            data_root=output_folder,
        )

    vis_train = Path(output_folder, "vis", "train")
    vis_val = Path(output_folder, "vis", "val")
    vis_train.mkdir(exist_ok=True, parents=True)

    show_colormaps(
        cmap_name=cmap_name,
        class_names=class_names,
        savepath=Path(output_folder, "colormap.png"),
    )

    img_train_folder = Path(output_folder, IMG_DIR, TRAIN_DIR)
    img_val_folder = Path(output_folder, IMG_DIR, VAL_DIR)
    ann_train_folder = Path(output_folder, ANN_DIR, TRAIN_DIR)
    ann_val_folder = Path(output_folder, ANN_DIR, VAL_DIR)

    visualize(
        ann_train_folder,
        img_train_folder,
        vis_train,
        ignore_substr_images_for_matching=RGB_EXT,
        ignore_substr_labels_for_matching=SEG_EXT,
        stride=vis_stride,
    )
    visualize(
        ann_val_folder,
        img_val_folder,
        vis_val,
        ignore_substr_images_for_matching=RGB_EXT,
        ignore_substr_labels_for_matching=SEG_EXT,
        stride=vis_stride,
    )
