def create_new_config(
    template_config_file,
    output_config_file,
    mean=None,
    std=None,
    classes=None,
    data_root=None,
):
    num_classes = len(classes)

    mean, std, classes, num_classes, data_root = [
        str(x) for x in (mean, std, classes, num_classes, data_root)
    ]

    with open(template_config_file, "r") as infile_h:
        with open(output_config_file, "w") as outfile_h:
            for line in infile_h.readlines():
                updated_line = (
                    line.replace("INSERT_MEAN", mean)
                    .replace("INSERT_STD", std)
                    .replace("INSERT_CLASSES", classes)
                    .replace("INSERT_NUM_CLASSES", num_classes)
                    .replace("INSERT_DATA_ROOT", '"' + data_root + '"')
                )
                outfile_h.write(updated_line)


if __name__ == "__main__":
    create_new_config(
        "configs/cityscapes_forests.py",
        "configs/cityscapes_forests_derived.py",
        [111.13859077733052, 117.59985508060974, 102.2959851538855],
        [73.30937007947439, 72.43582125151995, 65.35098998632455],
        ("ABCO", "CADE", "PILA", "PIPO", "PSME", "QUEV", "SNAG", "GROUND"),
        "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/species-class-seed-kernel/training_data/chips/",
    )
