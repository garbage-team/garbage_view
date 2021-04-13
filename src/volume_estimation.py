
import cv2
import numpy as np

if __name__ == '__main__':
    # Read image
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Select ROI
    fromCenter = False
    r = cv2.selectROI(im, fromCenter)

    # Crop image
    imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    # Display cropped image
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)
