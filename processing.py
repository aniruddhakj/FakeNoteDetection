'''This file contains the methods used to process the uploaded image'''

from cv2 import cv2

def readImage(file_name):
    '''Reads an image from the file system'''
    img = cv2.imread(file_name)
    return img


def resizeImage(image, scale):
    '''Resize image with fixed aspect ratio'''
    res = cv2.resize(image, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_AREA)
    return res


def imageToGray(image):
    '''Convert image to grayscale'''
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return img_gray


def medianBlur(image):
    '''Blurs an image using the median filter'''
    blurred_img = cv2.medianBlur(image, 3)
    return blurred_img


def adaptiveThresh(image):
    '''Applies adaptive thresholding to an image'''
    img_thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
    # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dsta
    return img_thresh