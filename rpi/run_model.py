from rpi.model_functions import *
from rpi.logger import *


def main():
    input_data = camera_capture()
    pred_depth = interpret_model(input_data)
    volume = depth_volume(pred_depth[0])
    log_depth_image(pred_depth[0], input_data[0])

    pred_img = cv2.rectangle(pred_depth[0], (38, 32), (38+146, 32+124), (0, 0, 255), 1)
    display_rgbd([input_data[0], pred_img])


if __name__ == '__main__':
    main()
