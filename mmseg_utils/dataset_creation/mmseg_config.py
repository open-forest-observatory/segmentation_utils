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
