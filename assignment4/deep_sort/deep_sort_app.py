# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import pickle
import cv2
import numpy as np

import matplotlib.pyplot as plt
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(detection_file):
    
    #hyperparameters
    max_cosine_distance = 0.2
    min_confidence = 0.6
    min_detection_height = 0
    nms_max_overlap = 0.3
    nn_budget = None

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    detection_file = np.load(detection_file)
    frame_idx =0;
    min_frame_idx = int(detection_file[:, 0].min())
    max_frame_idx = int(detection_file[:, 0].max())
    video = "/home/aman9875/Documents/deep_sort/output3.mp4"
    camera = cv2.VideoCapture(video)
    while True:
        (grabbed, frame) = camera.read()
        frame_idx += 1;
        print("Processing frame %05d" % frame_idx)
        # Load image and generate detections.
        detections = create_detections(detection_file, frame_idx-1, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
               continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        if grabbed == False:
            break

    results = np.asarray(results)
    results = results.astype(int)
    #print(results)
    np.savetxt("results.csv",results,fmt = "%d",delimiter = ',')

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.detection_file)
