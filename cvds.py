#!/usr/bin/env python2

import cv2
import sys
import numpy as np

#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('examples/demo-ds-still.mp4')
cap = cv2.VideoCapture('examples/demo-gbasp-shaky.mp4')
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

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sigma = cv2.getTrackbarPos(GAUSSIAN_LABEL, DEBUG_WINDOW)
    if sigma % 2 == 0:
        sigma = sigma + 1
    blur = cv2.GaussianBlur(frame, (sigma, sigma), 0)
    cv2.normalize(blur, blur, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if last_frame is None:
        last_frame = blur
        continue

    dt = cv2.absdiff(blur, last_frame)
    last_frame = blur

    # Apply binary thresholding to changed regions.
    # Only count significantly changed regions.
    delta_threshold = cv2.getTrackbarPos(OPTICAL_FLOW_LABEL, DEBUG_WINDOW)
    ret, thresh = cv2.threshold(dt, delta_threshold, 255, cv2.THRESH_BINARY)

    if mean is None:
        mean = thresh
        continue

    # Hard copy image for annotations
    annotations = image.copy()

    # Construct a new mean with the new threshold, decaying by a given constant.
    # This ensures that older changes are weighted less over time.
    # It also ensures that in the absence of optical flow, nothing changes.
    mean = cv2.addWeighted(mean, decay, thresh, 1, 0)
    cv2.normalize(mean, mean, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Threshold very small mean values from consideration.
    _, mean = cv2.threshold(mean, 15, 255, cv2.THRESH_TOZERO)

    # Normalize mean result for better responses.
    nmean = mean.copy()
    mean_integral = cv2.integral(nmean)

    # Find contours in original image.
    canny_thresh = cv2.getTrackbarPos(CANNY_LABEL, DEBUG_WINDOW)
    edges = cv2.Canny(blur, canny_thresh, canny_thresh * 2)
    dilated = cv2.dilate(edges, np.ones((10, 10)))

    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    bestScore = 0
    bestPoly = np.zeros((4, 1, 2), dtype='int32')

    total_area = image.shape[0] * image.shape[1]
    total_flow = mean_integral[-1,-1]

    cv2.drawContours(annotations, contours, 0, (255,0,0), 2)
    for c in contours:
        c_epsilon = cv2.getTrackbarPos(POLYFIT_LABEL, DEBUG_WINDOW)
        poly = cv2.approxPolyDP(c, c_epsilon, True)
        # Only consider quadrilaterals.
        if poly.shape[0] == 4:
            # Try and find the best fitting quadrilateral.
            cv2.drawContours(annotations, [poly], 0, (0,255,0), 3)
            area = cv2.contourArea(poly)
            x, y, w, h = cv2.boundingRect(poly) # Use bounding box plus integral image
                                                # to sum the optical flow.
            flow = mean_integral[y + h, x + w] \
                 - mean_integral[y + h, x] \
                 - mean_integral[y, x + w] \
                 + mean_integral[y, x]

            # TODO: is this the best result?
            #       it attempts to maximize flow and minimize area by the L1 norm
            #       maybe learn weights and train this with regression.

            # Ensure that we minimize regions that are too large.
            area_prop = area/total_area if total_area > 0 else 0
            area_score = 1 - 0.5 * area_prop

            # Percentage of image flow within this region.
            flow_score = float(flow)/float(total_flow) if total_flow > 0 else 0

            score = flow_score * area_score

            cv2.putText(annotations, "as: %.2f, fs: %.2f, ts: %.2f" %
                        (area_score, flow_score, score),
                        (x + 16, y + 16), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

            if score >= bestScore:
                bestScore = score
                bestPoly = poly

    cv2.drawContours(annotations, [bestPoly], 0, (0,0,255), 5)

    poly = np.array(bestPoly[:,0], dtype='Float32')

    # Ensure that the smallest normed sum is in the top left, and the largest
    # normed sum is in the bottom right.
    dst = np.array([
        [0, 0],
        [output_width - 1, 0],
        [0, output_height - 1],
        [output_width - 1, output_height - 1]],
        dtype = 'Float32')

    src = np.zeros((4, 2), dtype = 'Float32')
    src_sums = poly.sum(axis=1)
    src_diffs = np.diff(poly, axis=1)

    src[0] = poly[np.argmin(src_sums)]
    src[1] = poly[np.argmin(src_diffs)]
    src[2] = poly[np.argmax(src_diffs)]
    src[3] = poly[np.argmax(src_sums)]

    Mp = cv2.getPerspectiveTransform(src, dst)
    output = cv2.warpPerspective(image, Mp, (output_width, output_height))

    stage_output = np.concatenate([mean, dilated], axis=0)

    cv2.imshow("output", output)
    cv2.imshow("stage-output", stage_output)
    cv2.imshow(DEBUG_WINDOW, annotations)

    if cv2.waitKey(int(1000.0/30.0)) == 27: # esc
        break

cap.release()
cv2.destroyAllWindows()
