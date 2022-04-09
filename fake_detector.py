import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread('./files/Train/100_new/100.jpg',cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('./ground_truth/100_new/8.jpg',cv2.IMREAD_GRAYSCALE) # trainImage
# img2 = cv2.imread('./files/Train/100_new/3.jpg',cv2.IMREAD_GRAYSCALE) # trainImage


# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

print(len(kp1), len(kp2))
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
print(len(good))
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

#
# # Initiate ORB detector
# orb = cv2.ORB_create(nfeatures = 100000, scoreType=cv2.ORB_FAST_SCORE)
# # find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
#
# # cv2.imshow("1", img1)
# # cv2.imshow("2", img2)
# # cv2.waitKey(0)
# print(kp2)
#
# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # Match descriptors.
# matches = bf.match(des1,des2)
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# # Draw first 10 matches.
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()