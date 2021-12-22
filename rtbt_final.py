# Import all the packages that will be necessary
# Compartmentalized them for easier understanding
import numpy as np
import time
import cv2

import argparse
import imutils

from imutils.video import VideoStream
from collections import deque



# Over here, we intend to construct an argument parser
# and then those arguments are then parsed 
## We have added two arguments --
## One, where we read the webcam
## Two, where we read the video file - location needs to be specified
ap = argparse.ArgumentParser()
ap.add_argument('-b', '--buffer', default=128, type=int, help='maximum size of buffer')

# Will be commenting the video line as it is not necessary 
# and was deployed for testing purposes only. As in, to test 
# values for mask, trails, etc.
### ap.add_argument('-v', '--video', help='~/desktop/DSP/cricket.mp4')
args = vars(ap.parse_args())


# Here, we will be labeling the range of color for the ball
## In cricket, especially practice, the most common ball that is used
## is the tennis ball (easily available, cheap and not that injurious)

# Therefore, we will set a certain range for the model to detect the 
# color 'green'
## The range will be specified in the RGB space.

UpperRange = (62, 255, 255) # The color is very light (almost cyan)
LowerRange = (29, 88, 10)	# The color is slightly darker
points = deque(maxlen=args['buffer'])

## If any external video path was not specified,
## We need to initialize the program to use the webcam.
if args.get('video', True):
	videoStream = VideoStream(src=0).start() # Start the stream

## If the path was supplied and we want to use a pre-loaded video,
## We can do that by loading the file over here.
else:
	videoStream = cv2.VideoCapture(args['video'])

# Here, we set a buffer for about 3 seconds to let the webcam
# or the Video file to load and open
time.sleep(3.0)

# Now, we try to run an infinite loop for the videoStream
## The reason why we do so if we choose to have a pre-loaded video,
## We do not want the video to just end the detection to stop
## Instead, we want it to keep looping till we close the player
## And if it is a webcam interface (real-time), then we most certainly
## would not want it to end till we intervene
while True:
	# read every single frame
	frames = videoStream.read()

	# since we are focused only on the real-time part,
	# we will be using the videoStream and not videoCapture
	# videoCapture is only used when we are working with a video file
	frames = frames[1] if not args.get('video', True) else frames

	## should we choose to work with videoCapture and we are not
	## able to get any frames (the video has ended), then we just
	## break the loop
	if frames is None:
		break	# Go to the end of the program

	# Adjusting the frame to get the width, the mask and 
	# blur the rest of the frame for a higher FPS
	frames = imutils.resize(frames, width=600)
	blur = cv2.GaussianBlur(frames, (11, 11), 0)
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	

	# Now that is done, we apply a bit of processing 
	# techniques such as erosions and dialtions to the mask 
	## Note that the ball detected will be white in color in the mask.
	mask = cv2.inRange(hsv, LowerRange, UpperRange)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# Detecting the outline of the ball
	# Using the mask, we draw the outline on the ball and the countors
	# (outline) is then grabbed
	LContours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	## A center variable is created but right now, it is initialized to None
	center = None
	## grab the countors
	LContours = imutils.grab_contours(LContours)

	## If there was an outline detected, move forward to the next set
	## of execution where we verify if there was an outline detected
	if len(LContours) > 0:
		## Over here, we try to find the largest outline available in the frame
		## Situation when there are two balls available in the frame
		# and then we initialize that to maxContours
		maxContours = max(LContours, key=cv2.contourArea)
		## Initialize the (x, y) coordinates as well
		((x, y), radius) = cv2.minEnclosingCircle(maxContours)
		
		## Moments is used to store them in the memory
		M = cv2.moments(maxContours)
		center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

		## Check if the radius of the outline is less than 15 and move forward
		if radius > 15:
			## design a circle at the center of the outline and trace it forward
			## The color of the circle (for demo) was yellow but I will be using 
			## the color 'red'
			cv2.circle(frames, (int(x), int(y)), int(radius), (0, 0, 255), 2)
			## design another circle at the center of the red circle as the centroid
			## that one will be blue in color
			cv2.circle(frames, center, 5, (0, 0, 255), -1)

	## Keep updating the points in the queue of the blue blob
	points.appendleft(center)

	## run a loop to track the position of the ball using the blue centroid
	## that is at the center of the red outline
	for i in range(1, len(points)):
		## if there are no tracking points (there is no ball or trail in teh frame)
		## then ignore it
		if points[i - 1] is None or points[i] is None:
			continue
		## else, draw the thickness of the line
		thickness = int(np.sqrt(args['buffer'] / float(i + 1)) * 02.33)
		## draw a line of the same color as that of the centroid with the specified thickness
		## of the line
		cv2.line(frames, points[i - 1], points[i], (255, 0, 0), thickness)

	## display the frames on the screen
	key = cv2.waitKey(1) & 0xFF
	cv2.imshow('Frame', frames)

	## if the video file is being played, then in order to exit 
	## the loop, press the 'e' key
	if key == ord('e'):
		break

## if we are not using the video file, and are working with the webcam, then stop that
## stream too
# that can be done by pressing control + c on the command terminal
if args.get('video', True):
	videoStream.stop()

## else, keep continuing the stream and release it
else:
	videoStream.release()

## end the execution by destroying all the tabs that were generated and close the 
## execution. Often done by force quit or control + c
cv2.destroyAllWindows()