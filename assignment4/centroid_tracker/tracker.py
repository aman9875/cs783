# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from centroid_velocity_tracker import*

data = np.genfromtxt('Detectionfiles/detections1.csv',delimiter=',', dtype=np.float32)

default_confidence = 0.5

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker(15)

vs = cv2.VideoCapture("/home/ankusht/Documents/Study/CS783/Assignment4/Videos/output1.mp4")
writer = None

(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

frame_number = 0

COLORS = np.random.randint(0, 255, size=(5000, 3),
	dtype="uint8")


# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	(grabbed, frame) = vs.read()
	# frame = imutils.resize(frame, width=400)

	if not grabbed:
		break

	print("frame %d processed"%(frame_number))

	frame_number += 1

	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
	# blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
	# 	(104.0, 177.0, 123.0))
	# net.setInput(blob)
	# detections = net.forward()
	idx = data[:,0] == frame_number
	detections = data[idx]
	rects = []

	# loop over the detections
	for i in range(0, detections.shape[0]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
		if detections[i,6] > default_confidence:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			box = detections[i, 2:6]
			box[2:4] += box[0:2]
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		color = tuple(int(x) for x in COLORS[objectID%5000])
		# color = (0,255,0)
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)

	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("trackoutput1.avi", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	writer.write(frame)		


# do a bit of cleanup
vs.release()
writer.release()
