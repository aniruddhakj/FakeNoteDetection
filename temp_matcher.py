import cv2 # opencv binding
import imutils # Some methods of image processing
import numpy as np # numpy Perform numerical processing

kernel = np.ones((3, 3), np.float32) / 9

# Load template image , Convert grayscale , Detect edge
# Template matching using edges instead of the original image can greatly improve the accuracy of template matching .
def matcher(path,note):
    template = cv2.imread(path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.filter2D(template, -1, kernel)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    cv2.imshow("Template", template)

    # Traverse the image to match the template
    # Load image , Convert to grayscale , Initialize the bookkeeping variable used to track the matching area
    image = cv2.imread(note)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.filter2D(gray, -1, kernel)
    found = None
    # Traverse the image size
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # Zoom to image smaller than template , Then terminate
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        # Edge detection in the scaled grayscale image , Template matching
        # The image is calculated using exactly the same parameters as the template image Canny Edge representation ;
        # Use cv2.matchTemplate Apply template matching ;
        # cv2.minMaxLoc Get the relevant results and return a 4 Tuples , Which contains the minimum correlation value 、 Maximum correlation value 、 The minimum （x,y） Coordinates and maximum values （x,y） coordinate . We only deal with the maximum and （x,y）- Coordinates of interest , Therefore, only the maximum value is retained and the minimum value is discarded .
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    # Draw a bounding box in the detected area
        clone = np.dstack([edged, edged, edged])
        cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
        (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        # cv2.imshow("Visualize", clone)
        # cv2.waitKey(0)
        # If we find a new maximum correction value , Update bookkeeping variable values
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # Unpack the bookkeeping variables and based on the resizing ratio , Calculate the bounding box （x,y） coordinate
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    # Draw a bounding box on the detection result and display the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    print("Correlation Value:", maxVal, maxVal/((endX-startX)*(endY-startY)))
    # Need to perform thresholding on correlation value

def runner(note, den):
    
    for i in range(13):
        matcher(f"ground_truth/{den}/{i}.jpg",note)

runner('files/Train/100_new/100.jpg',"100_new")