from matplotlib import pyplot as plt
from cv2 import cv2

# read image as it is
def read_img(file_name):
    img = cv2.imread(file_name)
    return img


# resize image with fixed aspect ratio
def resize_img(image, scale):
    res = cv2.resize(image, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_AREA)
    return res


# convert image to grayscale
def img_to_gray(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return img_gray


# median blur
def median_blur(image):
    blurred_img = cv2.medianBlur(image, 3)
    return blurred_img


def adaptive_thresh(image):
    img_thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
    # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dsta
    return img_thresh