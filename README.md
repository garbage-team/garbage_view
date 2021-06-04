# garbage_view
Depth estimation for garbage containers in an industrial setting.

This repository contains code for setting up the model and its training pipeline. This includes loss functions, dataset generating and loading, image augmentation etc. This code is intended to run on a tensorflow compatible device, preferably using CUDA.

Using virtual normal loss by https://github.com/YvanYin/VNL_Monocular_Depth_Prediction

## Authors
Master's Thesis Project 2021 by

* Jonas Jungåker 
* Victor Hanefors

link coming soon

## Installation


Necessary dependencies:

* Python 3.8.x
* tensorflow 2.4.x
* tensorflow-addons
* matplotlib
* opencv-python
* numpy
* tensorflow-datasets

Recommended dependencies:

* Cuda/GPU support for tensorflow

## Data for training and validation
The model was first pretrained on [NYUDv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). The trained on data gathered using an Intel RealSense 3D-camera, code for that process can be found [here](https://github.com/garbage-team/realsense_camera). The dataset should contain a pairs of RGB and depth maps, stored together in a directory. The data can be seperated into several subfolder as long as each pair is in the same folder. The script will automaticaly split the input data randomly into training and validation, and saves them as seperate tfrecords.

To use NYUDv2 in training, simply run using (to be added).

To use self gathered dataset, first convert the data to a tfrecord (for easier handling) using (to be added). Then load the dataset using (to be added).

## Training
A complete training pipeline is setup in (to be added). Simply ensure the correct dataset is loaded, initialize a model or load a previous, verify the amount of epochs and/or any setting, and then run the script to start training. 

Our dataset for training and validation will be available for download, coming soon.


## Citation
Check out our thesis at: (link coming soon)

Cite our work with the following bib-file: (file coming soon)

### Acknowledgements
Thanks to Scania Smart Factory Lab for making this work possible

## License
Currently no license
