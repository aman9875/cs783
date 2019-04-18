# !pip install imutils
import os

import numpy as np
import argparse
import imutils
import time
import cv2
import os
import csv
from numpy import genfromtxt

# print(os.listdir("../input"))

# labelsPath = "../input/yolodataset/yolo_data/yolo-object-detection/yolo-coco/coco.names"
# LABELS = open(labelsPath).read().strip().split("\n")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

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

COLORS = np.random.randint(0, 255, size=(5000, 3),
	dtype="uint8")


frame_number = 0
data = genfromtxt('output.csv',delimiter = ',')
data.astype(int)
print(data[:,1])

while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	frame_number+=1
# 	if frame_number > 25:
# 		break
	print(frame_number)

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	print(frame.shape)	

	idx = data[:,0] == frame_number
	bbox = data[idx,3:7]
	bbox[:, 2:4] += bbox[:, 0:2]
	scores = data[idx, 7]
	classIds = data[idx,2]
	trackid = data[idx,1]
	for bb,s,clids,tid in zip(bbox,scores,classIds,trackid):
		# if frame_number == 1080:
		print("frame %d written"%(frame_number))
		# print("clids",clids)
		color = [int(c) for c in COLORS[int(tid)]]
		# print(color)
		cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 2)
		text = "{}: {} {:.4f}".format(CLASSES[int(clids)], tid,
				s)
		cv2.putText(frame, text, (int(bb[0]), int(bb[1]) - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("trackoutput1.avi", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	writer.write(frame)

writer.release()
vs.release()




