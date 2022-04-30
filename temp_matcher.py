import cv2  # opencv binding
import imutils  # Some methods of image processing
import numpy as np  # numpy Perform numerical processing
from Denomination.imageDisplay import display
import os

kernel = np.ones((3, 3), np.float32) / 9
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cannyLow, cannyHigh = 10, 150




def matcher(path, note):
    template = cv2.imread(path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # percent of original size (use 40% for complex flower design in 100 rupee note)
    scale_percent = 20
    width = int(template.shape[1] * scale_percent / 100)
    height = int(template.shape[0] * scale_percent / 100)
    dim = (width, height)
    template = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)

    template = cv2.filter2D(template, -1, kernel)
    (tH, tW) = template.shape[:2]

    image = cv2.imread(note)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.filter2D(gray, -1, kernel)
    found = None
    # Traverse the image size
    for scale in np.linspace(0.2, 2.0, 40)[::-1]:
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        # print(resized.shape, tH, tW)
        r = gray.shape[1] / float(resized.shape[1])
        # Zoom to image smaller than template , Then terminate
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        edged = resized
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        clone = np.dstack([edged, edged, edged])
        cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                      (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
            foundScale = 1/r

    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    resized = imutils.resize(gray, width=int(gray.shape[1] * foundScale))
    r = gray.shape[1] / float(resized.shape[1])
    edged = resized
    clone = np.dstack([edged, edged, edged])
    cv2.rectangle(clone, (int(startX*foundScale), int(startY*foundScale)),
                  (int(endX*foundScale), int(endY*foundScale)), (0, 0, 255), 2)


    print("Correlation Value:", found[0],
          found[0]/((endX-startX)*(endY-startY)))
    print("({},{}) to ({},{})".format(startX, startY, endX, endY))
    print("Scaling =", foundScale)

    return clone, template, found[0]


def runner(note, den):
    match_arr = []
    img_arr = []
    for f in os.listdir(str("./ground_truth/" + den + "/")):
        clone, template, conf = matcher("./ground_truth/" + den + "/"+f, note)
        img_arr.append((clone, template))
        match_arr.append(conf)
    return img_arr, match_arr

