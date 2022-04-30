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
    # train image
    train_img1 = cv2.imread(path)
    # cv2.imshow("color", train_img1)
    # display("color", train_img1)
    train_img = cv2.cvtColor(train_img1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("grayscale", train_img)
    # display("grayscale", train_img)
    edge_img = cv2.Canny(train_img, 30, 200)
    # cv2.imshow("edges", edge_img)

    # applying closing function
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("Closed", closed)

    # finding_contours
    (cnts, _) = cv2.findContours(closed.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    # for c in cnts:
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #     cv2.drawContours(train_img1, [approx], -1, (0, 255, 0), 2)
    # cv2.imshow("Output", train_img1)

    idx = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        idx += 1
        new_img = train_img1[y:y + h, x:x + w]
        cv2.imwrite("./tmp/cropped/crpd.png", new_img)
        cv2.rectangle(train_img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print("image cropped ")
        # cv2.imshow("box", train_img1)
        # cv2.waitKey(0)


# image = cv2.imread(training_set[14])
# gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edge_img = cv2.Canny(gray_img, 5, 200)
# cv2.imshow("gray", gray_img)
# cv2.imshow("edge", edge_img)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
# closed = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("Closed", closed)
#
# # finding_contours
# (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.03*peri, True)
#     if len(approx) == 4:
#         cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
# cv2.imshow("Output", image)
# cv2.waitKey(0)
