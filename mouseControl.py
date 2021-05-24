from pynput.mouse import Button, Controller
import pyscreenshot as ImageGrab


mouse = Controller()

# part of the screen
im = ImageGrab.grab(bbox=(1160, 325, 1500, 550))  # X1,Y1,X2,Y2

# save image file
im.save("box.png")
# use this function to get the exact coordinates of mouse press
while True:
    print('The current pointer position is {0}'.format(
        mouse.position))

# running the app is (40,515)
# starting the camera is (800,750)
# calibrating the camera is (1330,785) [same as measure, and next box]
# screen grab (1160-1500,325-550)


# go back to program (mouse click)
