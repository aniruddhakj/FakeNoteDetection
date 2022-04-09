from preprocessing.processing import readImage, resizeImage, imageToGray, medianBlur, adaptiveThresh, convertToBinary
# from imageDisplay import display
import os

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

train_path = 'files/Train/'
training_set = []
for f in os.listdir(train_path):
    if f == '.DS_Store':
        continue
    folder = (train_path + f)
    folder += '/'
    print(folder)
    for image in os.listdir(folder):
        if image.endswith(('.jpg', '.png', '.jpeg')):
            training_set.append(folder + image)

def f(start, end):

    for i in range(start, end):
        # train image
        train_img1 = cv2.imread(training_set[i])
        # cv2.imshow("color", train_img1)
        # display("color", train_img1)
        train_img = cv2.cvtColor(train_img1, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("grayscale", train_img)
        # display("grayscale", train_img)

        kernel = np.ones((3, 3), np.float32) / 9
        dst = cv2.filter2D(train_img, -1, kernel)
        cv2.imshow("blur", dst)

        edge_img = cv2.Canny(dst, 200, 250)
        cv2.imshow("edges", edge_img)

        # applying closing function
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Closed", closed)
        # closed = edge_img

        # finding_contours
        (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)#[:1]
        print(len(cnts))

        # for c in cnts:
        #     peri = cv2.arcLength(c, True)
        #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # cv2.drawContours(train_img1, [approx], -1, (0, 255, 0), 2)
        # cv2.imshow("Output", train_img1)

        # cv2.imwrite("./temp2/"+str(i)+".png", train_img1)
        ac = 0
        bc = 0
        idx = 0
        for c in cnts:
            if cv2.contourArea(c) <= 2:
                ac += 1
                continue
            else:
                bc += 1
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.00005*peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if True:#w > 150 and h > 150:
                idx += 1
                new_img = train_img1[y:y + h, x:x + w]
                # cv2.imwrite("./temp2/"+str(i)+"_"+str(idx) + '.png', new_img)
                cv2.rectangle(train_img1, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("box", train_img1)
    cv2.waitKey(0)
    print(ac, bc)

x=0
f(x, x+1)
# f(0, len(training_set))

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


