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

To use NYUDv2 in training, simply load the dataset using [load_nyudv2](https://github.com/garbage-team/garbage_view/blob/dev/src/data_loader.py#L89).

To use self gathered dataset, first convert the data to a tfrecord (for easier handling) using [write_tfrecord](https://github.com/garbage-team/garbage_view/blob/dev/src/data_loader.py#L118). Then load the dataset using [load_tfrecord](https://github.com/garbage-team/garbage_view/blob/dev/src/data_loader.py#L143).


## Guide: Setup model and train it.
An example of this process can be seen in [src/main.py](https://github.com/garbage-team/garbage_view/blob/a7f5a2ab0e1caa5d02306bf6ae7ecdccd26486c9/src/main.py#L12).

1. Generate a model, we recommend the softmax model [sm_model()](https://github.com/garbage-team/garbage_view/blob/333953227170c9cabfc815fad1eeeba0ff259250/src/model.py#L53) for easy implementation with the loss functions. 
2. Decide on the type of training you wish to do, which dataset, what parameters, which loss functions etc.
    * Dataset, learning rate and epochs is passed as parameters to the [training_loop()](https://github.com/garbage-team/garbage_view/blob/a7f5a2ab0e1caa5d02306bf6ae7ecdccd26486c9/src/main.py#L29).
    * Loss functions can be altered in the [custom_loss()](https://github.com/garbage-team/garbage_view/blob/a7f5a2ab0e1caa5d02306bf6ae7ecdccd26486c9/src/main.py#L70) function, and their respective weighing for the total loss.
3. Run [training_loop()](https://github.com/garbage-team/garbage_view/blob/a7f5a2ab0e1caa5d02306bf6ae7ecdccd26486c9/src/main.py#L29) and wait for the training to complete.
4. When training is completed, the model and training history will be saved.

(Optional)

* When the model has finished training, the model can tested by passing validation images to [test_model()](https://github.com/garbage-team/garbage_view/blob/a7f5a2ab0e1caa5d02306bf6ae7ecdccd26486c9/src/main.py#L119).
* The model history can be visualized using [plot_history()](https://github.com/garbage-team/garbage_view/blob/a7f5a2ab0e1caa5d02306bf6ae7ecdccd26486c9/src/image_utils.py#L257) to help determine if overfitting issues are present.


Our dataset for training and validation will be available for download, coming soon.


## Citation
Check out our thesis at: (link coming soon)

Cite our work with the following bib-file: (file coming soon)

### Acknowledgements
Thanks to Scania Smart Factory Lab for making this work possible

## License
Currently no license
