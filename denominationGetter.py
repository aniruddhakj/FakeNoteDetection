'''This file contains the method used to identify the denomination of the uploaded image'''

# from matplotlib import pyplot as plt
from cv2 import cv2
from os import listdir
from processing import read_img, resize_img, img_to_gray, median_blur, adaptive_thresh
from imageDisplay import display


def getDenomination(filePath):
    '''Function to identify the denomination of the uploaded image'''
    max_val = 8
    max_pt = -1
    max_kp = 0

    orb = cv2.ORB_create() # Create ORB object with default values

    test_img = read_img(filePath)

    original2 = resize_img(test_img, 0.4)
    # display('Input Image', original2)
    original1 = img_to_gray(original2)
    original3 = median_blur(original1)
    original = adaptive_thresh(original3)
    # display('Input Processed Image', original)
    # keypoints and descriptors
    (kp1, des1) = orb.detectAndCompute(test_img, None)

    train_path = 'files/Train/'
    training_set = []
    for f in listdir(train_path):
        if f == '.DS_Store':
            continue
        folder = (train_path + f)
        folder += '/'
        print(folder)
        for image in listdir(folder):
            if image.endswith(('.jpg', '.png', '.jpeg')):
                training_set.append(folder + image)

    for i in range(0, len(training_set)):
        # train image
        train_img1 = cv2.imread(training_set[i])
        train_img = cv2.cvtColor(train_img1, cv2.COLOR_BGR2GRAY)

        (kp2, des2) = orb.detectAndCompute(train_img, None)

        # brute force matcher
        bf = cv2.BFMatcher()
        all_matches = bf.knnMatch(des1, des2, k = 2)

        good = []
        for (m, n) in all_matches:
            if m.distance < 0.789 * n.distance:
                good.append([m])

        if len(good) > max_val:
            max_val = len(good)
            max_pt = i
            max_kp = kp2
        print(i, ' ', training_set[i], ' ', len(good))

    if max_val != 8:
        print(training_set[max_pt])
        print('good matches ', max_val)

        train_img = cv2.imread(training_set[max_pt])
        img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)

        note = str(training_set[max_pt])[12:-4]
        print('\nDetected note: ', note)
        # (plt.imshow(img3), plt.show())


    else:
        print('No Matches')
