import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from time import time
from src.image_utils import bins_to_depth


def depth_volume(depth):
    r = (38, 32, 146, 124) # TODO replace with points from identified borders
    depth = depth[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    volume = np.sum(depth)/np.sum(depth.size)
    print("Volume = " + str(volume))
    return volume


def camera_capture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype('float32')
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame


def interpret_model(input_data, model_path='lite_model_04-13.tflite'):
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
