import cv2
import numpy as np


def preproc_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    gray = clahe.apply(gray)

    gray = gray.reshape(gray.shape[0], gray.shape[1], 1)
    return gray


def preproc_morphology(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    gx = abs(gx)
    ret, img = cv2.threshold(gx, 100, 255, cv2.THRESH_BINARY)

    kernel = np.array([[1, 1, 1, 1]], dtype=np.uint8).T

    # open
    img = cv2.erode(img, kernel)
    img = cv2.dilate(img, kernel)

    kernel = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=np.uint8)

    # close
    img = cv2.dilate(img, kernel)
    img = cv2.erode(img, kernel)

    kernel = np.ones((7, 7), dtype=np.uint8)

    # open
    img = cv2.erode(img, kernel)
    img = cv2.dilate(img, kernel)

    return img


def preproc_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return bgr
