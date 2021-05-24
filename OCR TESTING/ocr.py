import pytesseract
import cv2
import numpy as np

'''
Using lets.traineddata compared to letsgodigital or ssd, ssd_int, 7seg because the former has the worst accuracy and the latter 3 read 7 as 1.
Will be using grayscale + erosion to get the best result. Only thing left is to physically adjust the device to get the most accurate reading  
'''

# get grayscale image


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # noise removal


def remove_noise(image):
    return cv2.medianBlur(image, 5)

# thresholding


def thresholding(image):
    # dilation
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# erosion


def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    # opening - erosion followed by dilation
    return cv2.erode(image, kernel, iterations=1)


def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    # canny edge detection
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def canny(image):
    return cv2.Canny(image, 100, 200)  # skew correction


def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated  # template matching


def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


for num in range(1, 10):
    filename = "testt("+str(num)+").jpg"
    print("Filename: "+filename)
    for i in range(9):
        img = cv2.imread(filename)
        img = get_grayscale(img)

        if(i == 1):
            img = remove_noise(img)
            print("Denoise")

        if(i == 2):
            img = thresholding(img)
            print("Thresholding")

        if(i == 3):
            img = dilate(img)
            print("Dilating")

        if(i == 4):
            img = erode(img)
            print("Erosion")

        if(i == 5):
            img = opening(img)
            print("Opening")

        if(i == 6):
            img = canny(img)
            print("Canny")

        if(i == 7):
            img = deskew(img)
            print("Deskew")

        cv2.imshow("Image", img)
        print(pytesseract.image_to_string(
            img, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789 -l lets'))
        cv2.waitKey(0)
