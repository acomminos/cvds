#!/usr/bin/env python2

import cv2
import sys
import numpy as np

#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('examples/demo-ds-still.mp4')
#cap = cv2.VideoCapture('examples/demo-gbasp-shaky.mp4')
#cap = cv2.VideoCapture('examples/demo-switch-complex.mp4')
cap = cv2.VideoCapture('examples/demo-gameboy-shaky.mp4')

last_frame = None
mean = None
decay = 0.9 # percentage of mean to retain each step

last_poly = None
poly_decay = 0.3 # percentage of last poly to keep each stage
                 # helps with smooth transitions

output_width = 512
output_height = 512

# Wishlist:
# - Hold good regions fixed over time.
# - Smoother interpolation.
#    - Outlier detection.

while True:
    ret, image = cap.read()
    if not ret:
        print "error"
        sys.exit(1)

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sigma = 19
    blur = cv2.GaussianBlur(frame, (sigma, sigma), 0)
    cv2.normalize(blur, blur, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if last_frame is None:
        last_frame = blur
        continue

    dt = cv2.absdiff(blur, last_frame)

    # Apply binary thresholding to changed regions.
    delta_threshold = 10 # Only count significantly changed regions.
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
    edges = cv2.Canny(blur, 25, 30)
    dilated = cv2.dilate(edges, np.ones((15, 15)))

    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bestRatio = 0
    bestPoly = None

    for c in contours:
        c_epsilon = 15
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
            ratio = flow/255 + 1/area

            if ratio > bestRatio:
                bestRatio = ratio
                bestPoly = poly

    if bestPoly is None:
        continue

    cv2.drawContours(annotations, [bestPoly], 0, (0,0,255), 5)

    if last_poly is None:
        last_poly = bestPoly.copy()
        continue

    # Consider the last poly when repositioning.
    last_poly = poly_decay * last_poly + (1 - poly_decay) * bestPoly
    #cv2.drawContours(annotations, [last_poly], 0, (255,0,0), 5)

    poly = np.array(last_poly[:,0], dtype='Float32')


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

    cv2.imshow("mean", nmean)
    cv2.imshow("edges", edges)
    cv2.imshow("dilated-edges", dilated)
    cv2.imshow("annotations", annotations)
    cv2.imshow("output", output)

    if cv2.waitKey(int(1000.0/60.0)) == 27: # esc
        break
    last_frame = blur

cap.release()
cv2.destroyAllWindows()
