from pynput.mouse import Button, Controller
import pyscreenshot as ImageGrab
import time
import pytesseract
import cv2


def findMousePosition():
    # use this function to get the exact coordinates of mouse press
    while True:
        print('The current pointer position is {0}'.format(mouse.position))


mouse = Controller()

"findMousePosition()"  # uncomment to find mouse position
# Press Button to open the application (40,515)
# Check where is the application located on the navigation panel
mouse.position = (40, 515)
mouse.press(Button.left)
mouse.release(Button.left)

# Wait for application to start (currently set to 5 seconds)
time.sleep(5)

# Press Button to start the camera (800,750)
mouse.position = (800, 750)
mouse.press(Button.left)
mouse.release(Button.left)

time.sleep(0.01)
# Calibrate the camera at (1330,785)
mouse.position = (1330, 785)
mouse.press(Button.left)
mouse.release(Button.left)

# Wait for camera to calibrate
time.sleep(5)

# Measure (1330,785)
mouse.position = (1330, 785)
mouse.press(Button.left)
mouse.release(Button.left)

# Wait for camera to calibrate
time.sleep(3)

# Screen Grab
im = ImageGrab.grab(bbox=(1160, 325, 1500, 550))  # X1,Y1,X2,Y2

# save image file
im.save("box.png")
# perform OCR
img = cv2.imread("box.png")
weight = pytesseract.image_to_string(
    img, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789 -l lets')
weight.replace(" ", "")
weight = int(weight)
# To get another Box (1330,785)
mouse.position = (1330, 785)
mouse.press(Button.left)
mouse.release(Button.left)

# Should delay for 0.1 seconds

# Go back to the applicaton (mouse click)
