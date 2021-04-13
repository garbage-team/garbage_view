import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from time import time
from math import pi
from src.config import cfg


def depth_volume(depth):
    #r = (38, 32, 146, 124)
    # TODO replace with points from identified borders
    # depth = depth[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    p = depth_to_xyz(depth)
    p = np.reshape(p, 224*224, 3)
    triangles = Delaunay(p[:, 0:2])
    print(triangles.simplicies.shape)
    print("Volume = " + str(volume))
    return volume


def depth_to_xyz(depth):
    x_size = depth.shape[1]
    y_size = depth.shape[0]

    x = np.asarray([i - (x_size // 2) for i in range(x_size)])  # [w,]
    x = np.tile(np.expand_dims(x, axis=0), (y_size, 1))         # [h, w]
    x = np.tan(cfg["webcam_h_fov"] * pi / 360) / (x_size / 2) * np.multiply(x, depth)

    y = np.asarray([i - (y_size // 2) for i in range(y_size)])  # [h,]
    y = np.tile(np.expand_dims(y, axis=-1), (1, x_size))        # [h, w]
    y = np.tan(cfg["webcam_h_fov"] * pi / 360) / (y_size / 2) * np.multiply(y, depth)

    z = depth  # TODO Might translate the point cloud along the z-axis, so that camera is not z=0

    x = np.expand_dims(x, -1)  # [b, h, w, 1]
    y = np.expand_dims(y, -1)  # [b, h, w, 1]
    z = np.expand_dims(z, -1)  # [b, h, w, 1]
    p = np.concatenate((x, y, z), axis=-1)  # [b, h, w, 3]

    return p


def camera_capture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype('float32')
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame


def interpret_model(input_data, model_path=cfg["tflite_model_path"]):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']

    tic = time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_depth = bins_to_depth(output_data)
    toc = time()
    print("Inference time: " + str(toc - tic))
    return pred_depth


def display_rgbd(images):
    # Displays a rgb image and a depth image side by side
    # Can take more than two images
    plt.figure(figsize=(15, len(images) * 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        if i < 1:
            plt.title("RGB image")
        else:
            plt.title("Depth image")
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()
    return None


def bins_to_depth(depth_bins):
    """
    Converts a bin tensor into a depth image
    Copy of src.image_utils bins_to_depth, but without tensorflow dependencies

    :param depth_bins: the depth bins in one_hot encoding, shape (b, h, w, c)
    the depth bins can also be passed as softmax bins of shape (b, h, w, c)
    :return: a depth image of shape (b, h, w) with type tf.float32
    """
    bin_interval = (np.log10(80) - np.log10(0.25)) / 150
    # the borders variable here holds the depth for each specific value of the one hot encoded bins
    borders = np.asarray([np.log10(0.25) + (bin_interval * (i + 0.5)) for i in range(150)])
    depth = np.matmul(depth_bins, borders)  # [b, h, w, (c] * [c), 1] -> [b, h, w, 1]
    depth = np.power(10., depth)
    return depth
