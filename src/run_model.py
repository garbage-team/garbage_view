import tflite_runtime.interpreter as tflite
import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2


def main():
    interpreter = tflite.Interpreter(model_path='lite_model_04-13.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype('float32')
    input_data = frame / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    tic = time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_depth = bins_to_depth(output_data)
    toc = time()
    print("Inference time: " + str(toc - tic))
    display_rgbd([input_data[0], pred_depth])


def bins_to_depth(depth_bins):
    """
    Converts a bin tensor into a depth image

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


if __name__ == '__main__':
    main()

