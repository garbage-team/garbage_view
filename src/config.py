# Config file for parameter values throughout the program
import numpy as np


# Consumers can access the config here
cfg = dict()

# Config for output and input of the model
cfg["input_size"] = (224, 224, 3)
cfg["depth_bins"] = 60
cfg["max_depth"] = 10.
cfg["min_depth"] = 0.25
cfg["bin_interval"] = (np.log10(cfg["max_depth"]) - np.log10(cfg["min_depth"])) / cfg["depth_bins"]

# Model encoder/decoder config
cfg["decoder_filters"] = [128, 64, 64, 64, 64] # 512, 256, 128, 96, 64
cfg["encoder_filters"] = [96, 144, 192, 576]
cfg["encoder_block_names"] = [
            "block_1_expand_relu",   # 112x112
            "block_3_expand_relu",   # 56x56
            "block_6_expand_relu",   # 28x28
            "block_13_expand_relu",  # 14x14
            "block_16_project",      # 7x7
    ]
cfg["model_bottleneck_channels"] = cfg["decoder_filters"][0]

# Loss function parameters
cfg["wcel_weights"] = [[np.exp(-0.2 * (i - j) ** 2) for i in range(cfg["depth_bins"])]
                       for j in np.arange(cfg["depth_bins"])]
cfg["vnl_sample_ratio"] = 0.15
cfg["vnl_discard_ratio"] = 0.25
cfg["fill_rate_loss_lim"] = [[-1, 1], [-1, 1], [1, 3]]

# Dataset parameters
cfg["data_focal_length"] = (1.0, 1.0)  # (focal_x, focal_y)
cfg["nyu_depth_path"] = 'D:/wsl/tensorflow_datasets'
cfg["tflite_model_path"] = "lite_model_04-13.tflite"

# FoV settings for different cameras.
# Logitech C930e (inference setup)
cfg["webcam_h_fov"] = 82.1
cfg["webcam_v_fov"] = 52.2
# Microsoft Kinect (NYUDV2)
cfg["kinect_h_fov"] = 62
cfg["kinect_v_fov"] = 48.6
# Intel RealSense D415
cfg["intel_h_fov"] = 69.4
cfg["intel_v_fov"] = 42.5
