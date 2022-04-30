import cv2  # opencv binding
import imutils  # Some methods of image processing
import numpy as np  # numpy Perform numerical processing
from Denomination.imageDisplay import display
import os

kernel = np.ones((3, 3), np.float32) / 9
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cannyLow, cannyHigh = 10, 150

# Load template image , Convert grayscale , Detect edge
# Template matching using edges instead of the original image can greatly improve the accuracy of template matching .


def matcher(path, note):
    template = cv2.imread(path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # percent of original size (use 40% for complex flower design in 100 rupee note)
    scale_percent = 20
    width = int(template.shape[1] * scale_percent / 100)
    height = int(template.shape[0] * scale_percent / 100)
    dim = (width, height)
    template = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)
    # cv2.imshow("template", template)
    template = cv2.filter2D(template, -1, kernel)
    # cv2.imshow("template", template)
    # template = cv2.Canny(template, cannyLow, cannyHigh)
    # template = cv2.morphologyEx(template, cv2.MORPH_CLOSE, kernel2)
    (tH, tW) = template.shape[:2]
    # cv2.imshow("Template", template)

    # Traverse the image to match the template
    # Load image , Convert to grayscale , Initialize the bookkeeping variable used to track the matching area
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
        # Edge detection in the scaled grayscale image , Template matching
        # The image is calculated using exactly the same parameters as the template image Canny Edge representation ;
        # Use cv2.matchTemplate Apply template matching ;
        # cv2.minMaxLoc Get the relevant results and return a 4 Tuples , Which contains the minimum correlation value 、 Maximum correlation value 、 The minimum （x,y） Coordinates and maximum values （x,y） coordinate . We only deal with the maximum and （x,y）- Coordinates of interest , Therefore, only the maximum value is retained and the minimum value is discarded .
        # edged = cv2.Canny(resized, cannyLow, cannyHigh)
        edged = resized
        # edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel2)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        # if maxVal > 0.7:
        #     print("dfsdsf", maxVal, maxLoc)

    # Draw a bounding box in the detected area
        clone = np.dstack([edged, edged, edged])
        cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                      (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        # cv2.imshow("Visualize", clone)
        # cv2.waitKey(0)
        # If we find a new maximum correction value , Update bookkeeping variable values
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
            foundScale = 1/r

    # Unpack the bookkeeping variables and based on the resizing ratio , Calculate the bounding box （x,y） coordinate
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    # Draw a bounding box on the detection result and display the image
    # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # display("Image", image)
    # cv2.waitKey(0)
    resized = imutils.resize(gray, width=int(gray.shape[1] * foundScale))
    r = gray.shape[1] / float(resized.shape[1])
    # edged = cv2.Canny(resized, cannyLow, cannyHigh)
    edged = resized
    # edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel2)
    # result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
    # (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    clone = np.dstack([edged, edged, edged])
    cv2.rectangle(clone, (int(startX*foundScale), int(startY*foundScale)),
                  (int(endX*foundScale), int(endY*foundScale)), (0, 0, 255), 2)
    # cv2.imshow("Visualize", clone)
    # cv2.waitKey(0)

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
    # for i in range(1+0, 1+9):
    #     num_matches += matcher(f"ground_truth/{den}/{i}.png",note)
    return img_arr, match_arr


# runner('files/Train/100_new/100.jpg',"100_new")
# runner('files/Test/india_100_2.jpg',"100_new")
# runner('files/Test/100.3.jpg',"100_new")
# runner('files/Test/19.jpg', "10_new")
# runner('files/Test/9.jpg', "10_new")
