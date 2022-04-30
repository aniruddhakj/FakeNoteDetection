import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import os

good_vals = []

# img1 is the input image and denomination it's denomination


def Matcher(img1_path, denomination):
    images = []
    global good_vals
    good_vals = []
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    for f in os.listdir(str("./ground_truth/" + denomination + "/")):

        img2 = cv2.imread(str("./ground_truth/" + denomination + "/" + f),
                          cv2.IMREAD_GRAYSCALE)
        images.append(SIFTMatcher(img1, img2))

    print(good_vals)
    avg = sum(good_vals)/len(good_vals)
    print("avg: ", avg)

    if (good_vals.count(0) + good_vals.count(1)) > 0.4*len(good_vals):
        print(" FAKE ")
        return (None, avg)
    else:

        if avg > len(good_vals):
            print("Seem legit boss")

    return (images, avg)


def SIFTMatcher(img1, img2):
    # initialize SIFT creater
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    try:
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        mi = 9999999999999
        v = 0

        for m, n in matches:
            v += 1
            if m.distance < 0.55*n.distance:
                good.append([m])
            mi = min(mi, abs(m.distance - n.distance))  # minimum distance

        good_vals.append(len(good))

        # print("***********************************************\nMinimum distance: {}".format(mi))
        # avg = sum(good)/len(good)

        # print("avg: ", avg)

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good,
                                  None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img3
    except Exception as e:
        print("SIFT Matcher failed", e)


def ORBMatcher(img1, img2):
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
        good = []
        for m in matches:
            if m.distance < 30:
                good.append(m)

        if (len(good) > 1):
            img3 = cv2.drawMatches(
                img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            return img3
        else:
            print("no good points in orb, trying sift for this security feature")
            return SIFTMatcher(img1, img2)

    except Exception as e:
        print("Orb Matcher failed, trying SIFT")
        return SIFTMatcher(img1, img2)


if __name__ == '__main__':
    img1 = cv2.imread('',
                      cv2.IMREAD_GRAYSCALE)          # queryImage
    # img2 = cv2.imread('./ground_truth/100_new/8.jpg',
    #                   cv2.IMREAD_GRAYSCALE)  # trainImage
    # img2 = cv2.imread('../ground_truth/100_new/8.png',
    #                   cv2.IMREAD_GRAYSCALE)  # trainImage testing for now
    # # img2 = cv2.imread('./files/Train/100_new/3.jpg',cv2.IMREAD_GRAYSCALE) # trainImage

    Matcher("../files/Train/100_new/100.jpg", "100_new")
