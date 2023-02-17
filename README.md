# MMSegmentation utils

The goal of this project is to provide helper functionality for semantic segmentation.
It is specifically targeted at the for the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library, but seeks to provide some utilities which could be more generally applicable.

# Setup

Install [poetry](https://python-poetry.org/docs/) to manage your dependencies. Then do `poetry install` to install the dependencies and this project in your current environment. Or first do `poetry shell` to create a new environment.

# mmsegmentation installation

Currently, this project depends on a docker image contained in this repo, [semantic slam](https://github.com/Kantor-Lab/semantic_slam). You need to build the docker image using the `build_docker.sh` script in that project, then you can run the `scripts/run_docker.sh` script in this project.

# Proceedure

The first step is to obtain a folder of images. Most of the data from SafeForest was logged as rosbags which are stored [here](https://github.com/Kantor-Lab/Safeforest_CMU_data_dvc).

If you are using a rosbag, one option is to use [my script](https://github.com/russelldj/DVC_ROS_datastore_scripts/blob/main/saving/images.py) which saves the timestamp and can geotag the images. Another is to use ROS's [`image_view` `image_saver`](http://wiki.ros.org/image_view).

Once you've obtained the data you can upload it to the platform of your choice for annotation. I'm using [VIAME](https://www.viametoolkit.org/) because it is free and open source. Furthermore, it has the capability to support active learning, which could be helpful later on. Labeling instructions can be found [here](https://docs.google.com/document/d/1bL3ECZmOwxqOrioqozR8EOY3NXcBvjb2PZbexVHa_Hk/edit?usp=sharing).
