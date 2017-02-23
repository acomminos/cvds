#!/usr/bin/env python2

import cv2
import sys
import numpy as np
from cvds import core
import getopt

def run_capture():
    if len(sys.argv) > 2:
        print "Usage: %s [path to video file]" % sys.argv[0]
        sys.exit(0)

    video_in = sys.argv[1] if len(sys.argv) == 2 else 0
    cap = cv2.VideoCapture(video_in)

    OUTPUT_WIDTH = 512
    OUTPUT_HEIGHT = 512

    DEBUG_WINDOW = "debug"
    cv2.namedWindow(DEBUG_WINDOW)

    CANNY_LABEL = "Canny"
    CANNY_DEFAULT = 20
    cv2.createTrackbar(CANNY_LABEL, DEBUG_WINDOW, CANNY_DEFAULT, 255, lambda _: None)

    GAUSSIAN_LABEL = "Gaussian"
    GAUSSIAN_DEFAULT = 3
    cv2.createTrackbar(GAUSSIAN_LABEL, DEBUG_WINDOW, GAUSSIAN_DEFAULT, 30, lambda _: None)

    FLOW_DECAY_LABEL = "Flow Decay Threshold"
    FLOW_DECAY_DEFAULT = 10
    cv2.createTrackbar(FLOW_DECAY_LABEL, DEBUG_WINDOW, FLOW_DECAY_DEFAULT, 100, lambda _: None)

    SCORE_DECAY_LABEL = "Score Decay Threshold"
    SCORE_DECAY_DEFAULT = 10
    cv2.createTrackbar(SCORE_DECAY_LABEL, DEBUG_WINDOW, SCORE_DECAY_DEFAULT, 100, lambda _: None)

    POLYFIT_LABEL = "Polygon Fit Epsilon"
    POLYFIT_DEFAULT = 50
    cv2.createTrackbar(POLYFIT_LABEL, DEBUG_WINDOW, POLYFIT_DEFAULT, 255, lambda _: None)

    acc = None
    while True:
        ret, image = cap.read()
        if not ret:
            print "Error opening up the capture stream."
            sys.exit(1)

        canny_thresh = cv2.getTrackbarPos(CANNY_LABEL, DEBUG_WINDOW)
        sigma = cv2.getTrackbarPos(GAUSSIAN_LABEL, DEBUG_WINDOW)
        flow_decay = float(cv2.getTrackbarPos(FLOW_DECAY_LABEL, DEBUG_WINDOW))/100.0
        score_decay = float(cv2.getTrackbarPos(SCORE_DECAY_LABEL, DEBUG_WINDOW))/100.0
        polyfit_epsilon = cv2.getTrackbarPos(POLYFIT_LABEL, DEBUG_WINDOW)

        annotations = image.copy()
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        poly, acc = core.find_flow_region(image_gray,
                sigma=sigma,
                edge_threshold=canny_thresh,
                fitting_error=polyfit_epsilon,
                flow_decay=flow_decay,
                score_decay=score_decay,
                acc=acc,
                annotations=annotations,
                debug=True)

        image_warped = core.warp_region(image, poly, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        cv2.imshow("output", image_warped)
        cv2.imshow(DEBUG_WINDOW, annotations)

        if cv2.waitKey(int(1000.0/30.0)) == 27: # esc
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_capture()
