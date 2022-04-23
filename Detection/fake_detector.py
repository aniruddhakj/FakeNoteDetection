import numpy as np
from cv2 import cv2
import matplotlib

import matplotlib.pyplot as plt
import os


# img1 is the input image and denomination it's denomination
def Matcher(img1_path, denomination):
    img1 = cv2.imread(img1_path)
    for f in os.listdir(str("./ground_truth/" + denomination + "/")):
        img2 = cv2.imread(str("./ground_truth/" + denomination + "/" + f),
                          cv2.IMREAD_GRAYSCALE)
        ORBMatcher(img1, img2)


def SIFTMatcher(img1, img2):
    # initialize SIFT creater
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(len(kp1), len(kp2))
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    try:
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        mi = 0

        # # Draw first 10 matches.
        # print("distance/ closest match", matches[0].distance, matches[1].distance)

        # img3 = cv2.drawMatches(
        #     img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3), plt.show()

        for m, n in matches:
            if m.distance < 0.55*n.distance:
                good.append([m])
            mi = min(mi, abs(m.distance - n.distance))  # minimum distance
        # cv.drawMatchesKnn expects list of lists as matches.
        print(len(good))
        print(mi)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good,
                                  None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3), plt.show()
    except:
        print("SIFT Matcher failed")


def ORBMatcher(img1, img2):
    # Initiate ORB detector
    try:
        orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        # Draw first 10 matches.
        img3 = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:7], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3), plt.show()
    except Exception as e:
        print("Orb Matcher failed, trying SIFT")


if __name__ == '__main__':
    img1 = cv2.imread('',
                      cv2.IMREAD_GRAYSCALE)          # queryImage
    # img2 = cv2.imread('./ground_truth/100_new/8.jpg',
    #                   cv2.IMREAD_GRAYSCALE)  # trainImage
    # img2 = cv2.imread('../ground_truth/100_new/8.png',
    #                   cv2.IMREAD_GRAYSCALE)  # trainImage testing for now
    # # img2 = cv2.imread('./files/Train/100_new/3.jpg',cv2.IMREAD_GRAYSCALE) # trainImage

    Matcher("../files/Train/100_new/100.jpg", "100_new")
