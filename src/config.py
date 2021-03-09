# Config file for parameter values throughout the program
import numpy as np


# Consumers can access the config here
cfg = dict()

# Config for output and input of the model
cfg["input_size"] = (224, 224, 3)
cfg["depth_bins"] = 150
cfg["max_depth"] = 80.
cfg["min_depth"] = 0.25
cfg["bin_interval"] = (np.log10(cfg["max_depth"]) - np.log10(cfg["min_depth"])) / cfg["depth_bins"]

# Model encoder/decoder config
cfg["decoder_filters"] = [512, 256, 128, 96, 64]
cfg["encoder_filters"] = [96, 144, 192, 576]
cfg["encoder_block_names"] = [
            "block_1_expand_relu",   # 112x112
            "block_3_expand_relu",   # 56x56
            "block_6_expand_relu",   # 28x28
            "block_13_expand_relu",  # 14x14
            "block_16_project",      # 7x7
    ]
cfg["model_bottleneck_channels"] = cfg["decoder_filters"][0]

# Loss function config
cfg["wcel_weights"] = [[np.exp(-0.2 * (i - j) ** 2) for i in range(cfg["depth_bins"])]
                       for j in np.arange(cfg["depth_bins"])]
