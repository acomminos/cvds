#!/usr/bin/env python2

import cv2
import numpy as np

def find_flow_region(image, sigma=5, edge_threshold=30, fitting_error=50,
                     flow_decay=0.1, score_decay=0.1, acc=None, annotations=None, debug=False):
    """Finds a quadrilateral region likely to contain an LCD display.
    This method attempts to find closed regions with four well-defined edges
    and a high amount of optical flow over the last few frames.

    image - The (grayscale) source image to be processed.
    sigma - The intensity of the gaussian to be applied for denoising.
    edge_threshold - The canny edge threshold used for finding initial edge regions.
    fitting_error - The threshold for screen fitting error. Increasing this may help detect screens.
    flow_decay - The amount prior frames' optical flow will be decreased each
                 subsequent frame. Increasing this can increase region stability
                 at the expense of tight motion tracking.
    score_decay - The amount the required score should decrease each frame
                  for the selection of a new candidate region.
                  Scores are normalized in [0, 1].
    acc - (optional) An opaque accumulator of optical flow data.
          If present, optical flow data will be used to prefer regions whose
          contents change frequently (like in an action video game).
          This will be updated in-place with the next frame's accumulator data.
    annotations - (optional) An optional output image upon which annotations
                  will be drawn related to area classification.

    Returns a tuple with a list of coordinates defining the quadrilateral, as well
    as the updated optical flow accumulator."""

    assert len(image.shape) == 2, "Image of depth 1 expected, got depth %d" % image.depth

    # We use the last frame to compute flow deltas.
    last_frame = np.zeros_like(image) if acc is None else acc['last_frame']

    # We define the "hot region" image as a weighted accumulator of regions that
    # have experienced optical flow recently.
    hot_region = np.zeros_like(image) if acc is None else acc['hot_region']

    # Grab ~97% of the distribution.
    filter_size = 2 * sigma + 1
    image_blur = cv2.GaussianBlur(image, (filter_size, filter_size), sigma)

    image_dt = cv2.absdiff(image_blur, last_frame)

    # Weight the latest deltas appropriately.
    cv2.addWeighted(image_dt, 1, hot_region, 1.0 - flow_decay, 0, hot_region)
    # Normalize the image so that if the new frame failed to introduce new
    # optical flow, we preserve the existing flow.
    cv2.normalize(hot_region, hot_region, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Threshold very small mean values from consideration.
    cv2.threshold(hot_region, 15, 255, cv2.THRESH_TOZERO, hot_region)
    # Compute an integral image for fast summation within candidate regions.
    hot_integral = cv2.integral(hot_region)

    image_edges = cv2.Canny(image_blur, edge_threshold, edge_threshold * 3)
    cv2.dilate(image_edges, np.ones((10, 10)), image_edges)

    contours, hierarchy = cv2.findContours(image_edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_poly = None
    best_score = None
    if acc is not None:
        best_poly = acc['last_poly']
        best_score = max(acc['last_score'] - score_decay, 0)
    else:
        best_poly = np.zeros((4, 1, 2), dtype='int32')
        best_score = 0

    total_area = image.shape[0] * image.shape[1]
    total_flow = hot_integral[-1,-1]

    if annotations is not None:
        cv2.drawContours(annotations, contours, 0, (255,0,0), 2)

    for c in contours:
        poly = cv2.approxPolyDP(c, fitting_error, True)
        # Only consider quadrilaterals.
        if poly.shape[0] == 4:
            # Try and find the best fitting quadrilateral.
            area = cv2.contourArea(poly)
            x, y, w, h = cv2.boundingRect(poly) # Use bounding box plus integral image
                                                # to sum the optical flow.
            flow = hot_integral[y + h, x + w] \
                 - hot_integral[y + h, x] \
                 - hot_integral[y, x + w] \
                 + hot_integral[y, x]

            # TODO: is this the best result?
            #       it attempts to maximize flow and minimize area by the L1 norm
            #       maybe learn weights and train this with regression.

            # Ensure that we minimize regions that are too large.
            area_prop = area/total_area if total_area > 0 else 0
            area_score = 1 - 0.5 * area_prop

            # Percentage of image flow within this region.
            flow_score = float(flow)/float(total_flow) if total_flow > 0 else 0

            score = flow_score * area_score

            if annotations is not None:
                TEXT_OFFSET = 16
                cv2.drawContours(annotations, [poly], 0, (0,255,0), 3)
                cv2.putText(annotations, "as: %.2f, fs: %.2f, ts: %.2f" %
                            (area_score, flow_score, score),
                            (x + TEXT_OFFSET, y + TEXT_OFFSET),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

            if score >= best_score:
                best_score = score
                best_poly = poly

    # TODO: move this to custom state object.
    next_acc = {
        'last_frame': image_blur,
        'hot_region': hot_region,
        'last_poly': best_poly,
        'last_score': best_score
    }

    if debug:
        # TODO: remove debug visualizations
        debug_output = np.vstack((image_blur, image_dt, image_edges, hot_region))
        debug_output_resized = cv2.resize(debug_output, (image.shape[0], image.shape[1]))
        cv2.imshow("cvds-debug", debug_output_resized)

    return (best_poly[:,0], next_acc)

def warp_region(image, poly, (output_width, output_height)):
    """Warps the quadrilateral defined by `poly` to a new image of the given size."""

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
    return cv2.warpPerspective(image, Mp, (output_width, output_height))
