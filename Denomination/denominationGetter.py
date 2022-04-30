'''This file contains the method used to identify the denomination of the uploaded image'''

# from matplotlib import pyplot as plt
from cv2 import cv2
from os import listdir
import matplotlib.pyplot as plt
from preprocessing.processing import readImage

# from preprocessing.processing import resizeImage, imageToGray, medianBlur, adaptiveThresh, convertToBinary
# from Denomination.imageDisplay import display


def getDenomination(filePath):
    '''Function to identify the denomination of the uploaded image'''
    max_val = 8
    max_pt = -1
    max_kp = 0

    # Create ORB object with default values
    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)
    test_img = readImage(filePath)

    (kp1, des1) = orb.detectAndCompute(test_img, None)

    train_path = 'files/Train/'
    training_set = []
    for f in listdir(train_path):
        if f == '.DS_Store':
            continue
        folder = (train_path + f)
        folder += '/'
        for image in listdir(folder):
            if image.endswith(('.jpg', '.png', '.jpeg')):
                training_set.append(folder + image)

    for i in range(0, len(training_set)):
        train_img1 = cv2.imread(training_set[i])
        train_img = cv2.cvtColor(train_img1, cv2.COLOR_BGR2GRAY)

        (kp2, des2) = orb.detectAndCompute(train_img, None)

        # Initialize the Brute Force Matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        good = []
        for m in matches:
            if m.distance < 50:
                good.append(m)
            else:
                break

        if len(good) > max_val:

            max_val = len(good)
            max_pt = i
            max_kp = kp2

        for m in matches:
            if m.distance < 50:
                good.append(m)

    if max_val != 8:
        train_img = cv2.imread(training_set[max_pt])
        note = str(training_set[max_pt])[12:-4]
        img3 = cv2.drawMatches(
            test_img, kp1, train_img, kp2, None, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return (note.split('/')[0], img3)

    else:
        print('No Matches')
        return ("", "")


if __name__ == '__main__':
    getDenomination('files/Test/100.jpg')
