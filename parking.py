####
# Python 2.7.9
# Raspberry pi 3
# Robo Car Project from ROBOLINK
####
# this program will try to follow an orange cone by using simple color detection
# adjust the trackbars to get a better filtered orange
####

import os
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

#this is for the pi camera to work
from picamera.array import PiRGBArray
from picamera import PiCamera

# time library for delays
import time

import RoPi_SerialCom as ropi

#this is open CV library
import cv2
import numpy as np #math libraries
import imutils #resizing the image frame

def nothing(x):
    pass

def filterColor(frame):
    """Make a color filter mask from trackbar
    """
    
    #collect all the trackbar positions
    h = cv2.getTrackbarPos('hue','Control Panel')
    s = cv2.getTrackbarPos('sat','Control Panel')
    v = cv2.getTrackbarPos('val','Control Panel')
    hr = cv2.getTrackbarPos('satRange', 'Control Panel')
    sr = cv2.getTrackbarPos('satRange', 'Control Panel')
    vr = cv2.getTrackbarPos('valRange', 'Control Panel')

    #use the trackbar positions to set
    #a boundary for the color filter
    hsvLower = (h-hr, s-sr, v-vr)
    hsvUpper = (h+hr, s+sr, v+vr)

    #turn into HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #mask will be a frame that has filtered the color we are looking for
    #that color will be within the hsvLower and hsvUpper constraints.
    mask = cv2.inRange(hsv, hsvLower, hsvUpper)

    return mask

def park_car(image):
    """ Detect the parking lanes and move the car into parking lot.
    Return:
        An image with detected lines drawn.
    """

    original_image = image

    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # removing noise with the basic morphological operation. Erosion
    #element = np.ones((5, 5))
    #processed_img = cv2.erode(processed_img, element)

    # smoothing the image
    processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)

    # edge detection with Canny algorithm
    processed_img = cv2.Canny(processed_img, threshold1=150, threshold2=300)

    min_len = cv2.getTrackbarPos('len', 'Control Panel')
    max_gap = cv2.getTrackbarPos('gap', 'Control Panel')
    threshold = cv2.getTrackbarPos('thresh', 'Control Panel')
    lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, threshold,
min_len, max_gap)

    # find the middle of the parking spot.
    x = []
    y = []
    try:
        for coords in lines:
            coords = coords[0]
            x += coords[0],coords[2]
            y += coords[1],coords[3]
            cv2.line(original_image, (coords[0],coords[1]),
(coords[2],coords[3]),(0,255,0),2)

        x = sorted(x)
        y = sorted(y)

        # ( min + max ) / 2
        x_mean = (x[0]+x[-1]) / 2
        y_mean = (y[0]+y[-1]) / 2

        if y_mean > 0:

            #if x pixel is below 40% of the frame width ex. x<(320*0.4=128)
            if(x_mean<(frame_width*(0.5-0.12))):
                ropi.moveRight()
                print("Right")
            elif (x_mean>(frame_width*(0.5+0.12))):
                ropi.moveLeft()
                print("Left")
            else:
                ropi.moveForwards()
                print("Go")
        else:
            ropi.moveStop()
            print("Stop")
    except:
        ropi.moveStop()
    return original_image


# Creating a window for later use
cv2.namedWindow('Control Panel')

# Creating track bar
#cv.CreateTrackbar(trackbarName, windowName, value, count, onChange)  None
cv2.createTrackbar('gap', 'Control Panel',5,1000,nothing)
cv2.createTrackbar('len', 'Control Panel',5,1000,nothing)
cv2.createTrackbar('thresh', 'Control Panel',5,255,nothing)
cv2.createTrackbar('hue', 'Control Panel',120,255,nothing)
cv2.createTrackbar('sat', 'Control Panel',100,255,nothing)
cv2.createTrackbar('val', 'Control Panel',125,255,nothing)
cv2.createTrackbar('hueRange', 'Control Panel',40,127,nothing)
cv2.createTrackbar('satRange', 'Control Panel',50,127,nothing)
cv2.createTrackbar('valRange', 'Control Panel',100,127,nothing)


# The tutorial to set up the PI camera comes from here
# http://www.pyimagesearch.com/2016/08/29/common-errors-using-the-raspberry-pi-camera-module/
# initialize the camera and grab a reference to the raw camera capture
#this is all to set up the pi camera
camera = PiCamera()
#resolution = (640, 480)
resolution = (320, 240)
#resolution = (160,128)
#resolution = (80,64)
#resolution = (48,32)
frame_width = (320)

#I use the lowest resolution to speed up the process

camera.resolution = resolution
#set frame rate to 32 frames per second
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=resolution)

#Allow the camera to warmup, if you dont sometimes the camera wont boot up
time.sleep(0.1)

ropi.speed(30)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr",
use_video_port=True):

    #grab intial time to measure how long
    #it takes to process the image
    e1 = cv2.getTickCount()

    # grab the raw NumPy array representing the image - this array
    # will be 3D, representing the width, height, and # of channels
    #convert the image into a numpy array
    frame = np.array(frame.array)

    #flips the frame if necessary change 0 to 1 or 2..
    frame = cv2.flip(frame,0)

    mask = filterColor(frame)

    #this "res" frame pieces together 2 the mask frame and the original frame
    res = cv2.bitwise_and(frame,frame, mask = mask)

    frame_ = park_car(res)

    cv2.imshow("frame", frame_)
    cv2.imshow("res", res)

    #the key that is clicked save it as variable key
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    if key == ord("Q"):
        break

    e2 = cv2.getTickCount()
    time = (e2 - e1)/cv2.getTickFrequency()
    # print "milliSeconds" , time*1000

# clear the stream in preparation for the next frame
rawCapture.truncate(0)
cv2.destroyAllWindows()

#just in case the robot is still moving
ropi.moveStop()