from preprocessing.processing import readImage, resizeImage, imageToGray, medianBlur, adaptiveThresh, convertToBinary
# from imageDisplay import display
import os

import matplotlib.pyplot as plt
from cv2 import cv2


def initialize():
    train_path = 'tmp/'
    training_set = []
    for f in os.listdir(train_path):
        if f == '.DS_Store':
            continue
        folder = (train_path + f)
        for image in os.listdir(folder):
            if image.endswith(('.jpg', '.png', '.jpeg')):
                training_set.append(folder + image)
    f(training_set)


def f(path):
    train_img1 = cv2.imread(path)
    train_img = cv2.cvtColor(train_img1, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(train_img, 30, 200)

    # applying closing function
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("Closed", closed)

    # finding_contours
    (cnts, _) = cv2.findContours(closed.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]



    idx = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        idx += 1
        new_img = train_img1[y:y + h, x:x + w]
        cv2.imwrite("./tmp/cropped/crpd.png", new_img)
        cv2.rectangle(train_img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print("image cropped ")




