# cvds #

![](https://raw.githubusercontent.com/acomminos/cvds/master/demo.gif)

cvds is an experimental library written using OpenCV and Python for detecting and warping recordings of LCD displays into higher quality, undistorted footage.

The initial motivation behind this project was to develop a real-time solution for portable console recording.

While there exist fundamental limitations of this approach (such as the nyquist rate in many cases, both spatially and temporally), the goal of cvds is to provide a tool to improve the quality of screen recordings with good enough consistency and quality to warrant its use as a viable postprocessing step.

## How it Works ##

cvds scores quadrilateral regions found using OpenCV's implementation of Douglas-Peucker using a linearly weighted function of area and prior optical flow. The core assumption made by cvds is the quadrilateral region in the scene with the highest amount of optical flow over the last few frames is likely to be a display.

This assumption is appropriate for cases where the target display is the focus of the scene.

cvds provides utilities for warping this region using a reverse perspective transform into a generic rectangular image approximating the screen's original content.
