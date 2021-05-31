# Importing KIVY modules
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import ScreenManager, Screen

# Import RealSense, OpenCV and NumPy
import pyrealsense2 as rs
import pytesseract
import time
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from collections import defaultdict
from CV_Intel.realsense_device_manager import DeviceManager
from CV_Intel.calibration_kabsch import PoseEstimation
from CV_Intel.helper_functions import get_boundary_corners_2D
from CV_Intel.measurement_task import calculate_boundingbox_points, calculate_cumulative_pointcloud, visualise_measurements

# Setting up Different Screens
sm = ScreenManager()
screens = [Screen(name='Start Page'), Screen(
    name='Calibration'), Screen(name='DIM Measurement'), Screen(name='Unable to sense checkerboard')]

# Parametes that can be tweaked
chessboard_width = 6  # squares
chessboard_height = 9 	# squares
square_size = 0.0253  # meters


# Global Variables to store the information required
namev = ""
lengthv = 0
widthv = 0
heightv = 0
weightv = 0
df = pd.DataFrame(columns=["SKU", "Length",
                  "Width", "Height", "Weight"])

# onStart Window Size
Window.size = (900, 700)
# Background Colour
Window.clearcolor = (110/255.0, 147/255.0, 222/255.0, 1)

# Button Callback Methods


def measuref(instance):

    # Computer Vision Method
    computerVisionMethod()

    # LIDAR Method
    lidarMethod()

    # OCR Method
    weight = ocrRead()

    # Updates the current stored value
    lengthv = int(length*1000)
    widthv = int(width*1000)
    heightv = int(height*1000)
    weightv = weight

    # Swaps the Text for Display
    lengthInput.text = str(lengthv)
    widthInput.text = str(widthv)
    heightInput.text = str(heightv)
    weightInput.text = str(weightv)


def submitf(instance):
    global df
    # Take in the inputs and add it to the DF
    temp_df = pd.DataFrame([[nameInput.text, lengthInput.text, widthInput.text, heightInput.text, weightInput.text]], columns=["SKU", "Length",
                                                                                                                               "Width", "Height", "Weight"])
    print(temp_df)

    if df.empty:
        df = temp_df
    else:
        df = df.append(temp_df)

    print(df)

    # Refresh List View


def deletef(instance):
    print("Delete a particular entry")


def createcsvf(instance):
    global df
    # Create csv from DataFrame
    date = datetime.now()

    month = date.strftime("%m")
    day = date.strftime("%d")
    year = date.strftime("%y")
    hour = date.strftime("%H")
    minute = date.strftime("%M")

    fileName = day+"-"+month+"-"+year+"--"+hour+minute+".csv"
    df.to_csv(fileName, index=False, header=True)

    df = pd.DataFrame(columns=["SKU", "Length",
                               "Width", "Height", "Weight"])


def calibratef(instance):

    # Define some constants
    resolution_width = 1280  # pixels
    resolution_height = 720  # pixels
    frame_rate = 15  # fps
    dispose_frames_for_stablisation = 30  # frames

    try:
        # Enable the streams from all the intel realsense devices
        sm.switch_to(screens[1])
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(
            rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
        rs_config.enable_stream(
            rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

        # Use the device manager class to enable the devices and get the frames
        global device_manager
        device_manager = DeviceManager(rs.context(), rs_config)
        device_manager.enable_all_devices()

        # Allow some frames for the auto-exposure controller to stablise
        for frame in range(dispose_frames_for_stablisation):
            frames = device_manager.poll_frames()

        assert(len(device_manager._available_devices) > 0)
        """
		1: Calibration
		Calibrate all the available devices to the world co-ordinates.
		For this purpose, a chessboard printout for use with opencv based calibration process is needed.

		"""
        # Get the intrinsics of the realsense device
        global intrinsics_devices
        intrinsics_devices = device_manager.get_device_intrinsics(frames)

        # Set the chessboard parameters for calibration
        chessboard_params = [chessboard_height, chessboard_width, square_size]

        # Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
        calibrated_device_count = 0
        while calibrated_device_count < len(device_manager._available_devices):
            frames = device_manager.poll_frames()
            pose_estimator = PoseEstimation(
                frames, intrinsics_devices, chessboard_params)
            transformation_result_kabsch = pose_estimator.perform_pose_estimation()
            object_point = pose_estimator.get_chessboard_corners_in3d()
            calibrated_device_count = 0
            for device in device_manager._available_devices:
                if not transformation_result_kabsch[device][0]:
                    # add the label to place chessboard
                    print(
                        "Place the chessboard on the plane where the object needs to be detected..")
                    sm.switch_to(screens[3])

                else:
                    calibrated_device_count += 1

        # Save the transformation object for all devices in an array to use for measurements
        global transformation_devices
        transformation_devices = {}
        chessboard_points_cumulative_3d = np.array([-1, -1, -1]).transpose()
        for device in device_manager._available_devices:
            transformation_devices[device] = transformation_result_kabsch[device][1].inverse(
            )
            points3D = object_point[device][2][:, object_point[device][3]]
            points3D = transformation_devices[device].apply_transformation(
                points3D)
            chessboard_points_cumulative_3d = np.column_stack(
                (chessboard_points_cumulative_3d, points3D))

        # Extract the bounds between which the object's dimensions are needed
        # It is necessary for this demo that the object's length and breath is smaller than that of the chessboard
        chessboard_points_cumulative_3d = np.delete(
            chessboard_points_cumulative_3d, 0, 1)
        global roi_2D
        roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)

        print("Calibration completed... \nPlace the box in the field of view of the devices...")

        """
                2: Measurement and display
                Measure the dimension of the object using depth maps from multiple RealSense devices
                The information from Phase 1 will be used here

                """

        # Enable the emitter of the devices
        device_manager.enable_emitter(True)

        # Load the JSON settings file in order to enable High Accuracy preset for the realsense
        device_manager.load_settings_json("./HighResHighAccuracyPreset.json")

        # Get the extrinsics of the device to be used later
        extrinsics_devices = device_manager.get_depth_to_color_extrinsics(
            frames)

        # Get the calibration info as a dictionary to help with display of the measurements onto the color image instead of infra red image
        global calibration_info_devices
        calibration_info_devices = defaultdict(list)
        for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
            for key, value in calibration_info.items():
                calibration_info_devices[key].append(value)

        # go to second screen
        sm.switch_to(screens[2])
    except:
        print("The program was interupted by the user. Closing the program...")
        # change screen
        sm.switch_to(screens[3])

    # run calibration code

# function taht is called when you type stuff in the text input


def onNameType(instance, value):
    global namev
    namev = value
    print('The widget', instance, 'have:', value)


def onLengthType(instance, value):
    global lengthv
    lengthv = value
    print('The widget', instance, 'have:', value)


def onWidthType(instance, value):
    global widthv
    widthv = value
    print('The widget', instance, 'have:', value)


def onHeightType(instance, value):
    global heightv
    heightv = value
    print('The widget', instance, 'have:', value)


def onWeightType(instance, value):
    global weightv
    weightv = value
    print('The widget', instance, 'have:', value)


# Styling of the GUI

layout = FloatLayout(size=(900, 700))
calibrationlayout = FloatLayout(size=(900, 700))
startlayout = FloatLayout(size=(900, 700))
errorlayout = FloatLayout(size=(900, 700))


# Text Inputs
nameInput = TextInput(text='', size_hint=(.2, .05),
                      pos_hint={'x': .45, 'y': .85})

lengthInput = TextInput(text=str(lengthv), multiline=False, size_hint=(.2, .05),
                        pos_hint={'x': .15, 'y': .725})
widthInput = TextInput(text=str(widthv), multiline=False, size_hint=(.2, .05),
                       pos_hint={'x': .45, 'y': .725})
heightInput = TextInput(text=str(heightv), multiline=False, size_hint=(.2, .05),
                        pos_hint={'x': .75, 'y': .725})
weightInput = TextInput(text=str(weightv), multiline=False, size_hint=(.2, .05),
                        pos_hint={'x': .25, 'y': .625})


# Labels
title = Label(text="Dimensioning System for Bollore Operators",
              size_hint=(.5, .05),
              pos_hint={'x': .25, 'y': .95})
name = Label(text="SKU Name:",
             size_hint=(.2, .05),
             pos_hint={'x': .25, 'y': .85})
length = Label(text="Length (cm):",
               size_hint=(.1, .05),
               pos_hint={'x': .05, 'y': .725})
width = Label(text="Width (cm):",
              size_hint=(.1, .05),
              pos_hint={'x': .35, 'y': .725})
height = Label(text="Height (cm):",
               size_hint=(.1, .05),
               pos_hint={'x': .65, 'y': .725})
weight = Label(text="Weight (g):",
               size_hint=(.1, .05),
               pos_hint={'x': .15, 'y': .625})
welcomeTitle = Label(text="Welcome to DIY DIM System\nPress start to begin Calibration",
                     size_hint=(.5, .05),
                     pos_hint={'x': .25, 'y': .75})
calibrationTitle = Label(text="Calibrating...",
                         size_hint=(.5, .2),
                         pos_hint={'x': .25, 'y': .4})
errorTitle = Label(text="Unable to find the checkerboard\nPlease close the program and try again\nMake sure the checkerboard is within the view of the camera",
                   size_hint=(.5, .2),
                   pos_hint={'x': .25, 'y': .4})


# Buttons
measure = Button(
    text='Measure',
    size_hint=(.4, .05),
    pos_hint={'x': .3, 'y': .79})
submit = Button(
    text='Submit',
    size_hint=(.4, .05),
    pos_hint={'x': .3, 'y': .56})
delete = Button(
    text='Delete',
    size_hint=(.5, .05),
    pos_hint={'x': .2, 'y': .2})
createcsv = Button(
    text='Create CSV',
    size_hint=(.2, .05),
    pos_hint={'x': .75, 'y': .2})
calibrate = Button(
    text='Start',
    size_hint=(.3, .2),
    pos_hint={'x': .35, 'y': .4})

# Binding onPress Method
measure.bind(on_press=measuref)
submit.bind(on_press=submitf)
delete.bind(on_press=deletef)
createcsv.bind(on_press=createcsvf)
calibrate.bind(on_press=calibratef)

# Binding onType Methods
nameInput.bind(text=onNameType)
lengthInput.bind(text=onLengthType)
widthInput.bind(text=onWidthType)
heightInput.bind(text=onHeightType)
weightInput.bind(text=onWeightType)

# Adding List View
listview = ScrollView(size_hint=(0.5, 0.5), size=(Window.width, Window.height))
listview.add_widget(delete)


# Adding the Widgets to the Layout
layout.add_widget(title)
layout.add_widget(measure)
layout.add_widget(submit)
layout.add_widget(createcsv)
layout.add_widget(name)
layout.add_widget(length)
layout.add_widget(width)
layout.add_widget(height)
layout.add_widget(weight)
layout.add_widget(nameInput)
layout.add_widget(lengthInput)
layout.add_widget(widthInput)
layout.add_widget(heightInput)
layout.add_widget(weightInput)
layout.add_widget(listview)
startlayout.add_widget(calibrate)
startlayout.add_widget(welcomeTitle)
calibrationlayout.add_widget(calibrationTitle)
errorlayout.add_widget(errorTitle)


screens[0].add_widget(layout)
screens[1].add_widget(calibrationlayout)
screens[2].add_widget(startlayout)
screens[3].add_widget(errorlayout)

sm.add_widget(screens[0])
sm.add_widget(screens[1])
sm.add_widget(screens[2])
sm.add_widget(screens[3])

# Building the Layout


class DIMSoftware(App):
    def build(self):
        return sm


# Deploying the Software
DIMSoftware().run()
device_manager.disable_streams()


# Erosion Method for OCR
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    # opening - erosion followed by dilation
    return cv2.erode(image, kernel, iterations=1)

# Dimension Measurement Methods


def computerVisionMethod():
    # Get the values from the camera
    frames_devices = device_manager.poll_frames()

    # Calculate the pointcloud using the depth frames from all the devices
    point_cloud = calculate_cumulative_pointcloud(
        frames_devices, calibration_info_devices, roi_2D)

    # Get the bounding box for the pointcloud in image coordinates of the color imager
    bounding_box_points_color_image, length, width, height = calculate_boundingbox_points(
        point_cloud, calibration_info_devices)


def lidarMethod():
    # add the new code after testing from mouseControl.py
    pass


def ocrRead():
    # take a picture
    camera_port = 0  # can toggle the value to switch between the laptop camera and the webcam
    camera = cv2.VideoCapture(camera_port)
    # If you don't wait, the image will be dark
    time.sleep(0.1)
    return_value, image = camera.read()
    cv2.imwrite("picture.png", image)
    # so that others can use the camera as soon as possible
    del(camera)

    # crop the picture
    crop_img = cv2.imread("picture.png")
    # to be done manually after the mount has been set
    crop_img = crop_img[0:367, 400:872]

    # perform OCR read
    img = erode(crop_img)
    weight = pytesseract.image_to_string(
        img, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789 -l lets')
    weight.replace(" ", "")
    weight = int(weight)

    return weight
