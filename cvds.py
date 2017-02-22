#!/usr/bin/env python2

import cv2
import sys
import numpy as np

from cvds_core import CVDSEngine

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('examples/demo-ds-still.mp4')
#cap = cv2.VideoCapture('examples/demo-gbasp-shaky.mp4')
#cap = cv2.VideoCapture('examples/demo-switch-complex.mp4')
#cap = cv2.VideoCapture('examples/demo-gameboy-shaky.mp4')
#cap = cv2.VideoCapture('examples/demo-gbc-shaky.mp4')

last_frame = None
mean = None
decay = 0.9 # percentage of mean to retain each step

output_width = 512
output_height = 512

# Wishlist:
# - Hold good regions fixed over time.
# - Smoother interpolation.
#    - Outlier detection.

DEBUG_WINDOW = "debug"

cv2.namedWindow(DEBUG_WINDOW)

CANNY_LABEL = "Canny"
CANNY_DEFAULT = 30
cv2.createTrackbar(CANNY_LABEL, DEBUG_WINDOW, CANNY_DEFAULT, 255, lambda _: None)

GAUSSIAN_LABEL = "Gaussian"
GAUSSIAN_DEFAULT = 5
cv2.createTrackbar(GAUSSIAN_LABEL, DEBUG_WINDOW, GAUSSIAN_DEFAULT, 30, lambda _: None)

OPTICAL_FLOW_LABEL = "Optical Flow Threshold"
OPTICAL_FLOW_DEFAULT = 10
cv2.createTrackbar(OPTICAL_FLOW_LABEL, DEBUG_WINDOW, OPTICAL_FLOW_DEFAULT, 255, lambda _: None)

POLYFIT_LABEL = "Polygon Fit Epsilon"
POLYFIT_DEFAULT = 50
cv2.createTrackbar(POLYFIT_LABEL, DEBUG_WINDOW, POLYFIT_DEFAULT, 255, lambda _: None)

while True:
    ret, image = cap.read()
    if not ret:
        print "Error:", ret
        sys.exit(1)


    cv2.imshow("output", output)
    cv2.imshow("stage-output", stage_output)
    cv2.imshow(DEBUG_WINDOW, annotations)

    if cv2.waitKey(int(1000.0/30.0)) == 27: # esc
        break

cap.release()
cv2.destroyAllWindows()
