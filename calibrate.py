import cv2
import numpy as np
def calibrateCamera():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:8].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    cam = cv2.VideoCapture(0)
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (6, 8), None)
        if not ret:
            print("Failed to find corners")
        else:
            # If found, add object points, image points (after refining them)
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img_corners = cv2.drawChessboardCorners(img, (6, 8), corners2, ret)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))


            # Save settings for undistorting other files
            try:
                np.savetxt('roi.txt', roi, delimiter=',')
                np.savetxt('cam_mtx.txt', newcameramtx, delimiter=',')
                np.savetxt('dist.txt',dist, delimiter=',')
            except:
                print("Could not save calibration files, please run calibration again.")
            else:
                print("Calibration data saved successfully!")


            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            cv2.imwrite('calibresult.png', dst)
            cv2.imshow('Orignal',img)
            cv2.imshow('Undistorted',dst)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


def main():
    calibrateCamera()


if __name__== "__main__":
    main()