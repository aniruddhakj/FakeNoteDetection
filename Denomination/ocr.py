from pytesseract import *
import cv2
import numpy as np


pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
kernel = np.ones((5, 5), np.float32) / 25
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def getDenominationCharacters(filePath):
    image = cv2.imread(filePath)
    scale_percent = 100  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resize = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    # gray = 255-gray
    smooth = cv2.filter2D(gray, -1, kernel)
    smooth = cv2.GaussianBlur(gray, (7,7), 0)
    canny = cv2.Canny(smooth, 250, 255)
    morph = cv2.dilate(canny, kernel2, iterations=1)
    cv2.imshow("Image", morph)
    cv2.waitKey(0)

    results = pytesseract.image_to_data(morph, output_type=Output.DICT)

    for i in range(0, len(results["text"])):
        text = results["text"][i]
        print(text)




if __name__ == '__main__':
    getDenominationCharacters('../files/Test/100.3.jpg')