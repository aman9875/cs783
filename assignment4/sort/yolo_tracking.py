
import sys
sys.path.insert(0,"/home/aman9875/Documents/yolo-object-detection/")
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from sort import *

mot_tracker = Sort()
labelsPath = "/home/aman9875/Documents/yolo-object-detection/yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = "/home/aman9875/Documents/yolo-object-detection/yolo-coco/yolov3.weights"
configPath = "/home/aman9875/Documents/yolo-object-detection/yolo-coco/yolov3.cfg"

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture("/home/aman9875/Documents/yolo-object-detection/videos/test1.mp4")
writer = None
(W, H) = (None, None)

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
		break

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	detections = []
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.5,
		0.3)

	for i in idxs.flatten():
		(x,y) = (boxes[i][0],boxes[i][1])
		(w,h) = (boxes[i][2],boxes[i][3])
		confidence = confidences[i]
		detections.append([x,y,x+w,y+h,confidence,classIDs[i]])	
	detections = np.asarray(detections)
	track_bbs_ids = mot_tracker.update(detections)
	
	track_bbs_ids = track_bbs_ids.astype(int)
	if len(track_bbs_ids) > 0:
		for x1,y1,x2,y2,obj_id,cls_pred in track_bbs_ids:
			color = [int(c) for c in COLORS[int(obj_id)%len(COLORS)]]
			cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
			#print("%s %d" %(LABELS[cls_pred],obj_id))
			text = LABELS[cls_pred] + " " + str(obj_id)
			#print(text)
			cv2.putText(frame,text,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# cv2.imshow('tracking',frame)
	# k = cv2.waitKey(0)
	# if k == 27:         # wait for ESC key to exit
	# 	cv2.destroyAllWindows()
	# check if the video writer is None

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output/test1.avi", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	writer.write(frame)
            
print("[INFO] cleaning up...")
writer.release()
vs.release()