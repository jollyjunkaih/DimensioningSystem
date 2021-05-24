import cv2

img = cv2.imread("test0.jpg")
crop_img = img[0:367, 400:872]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
