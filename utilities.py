from __future__ import print_function
from scipy.spatial import distance as dist
import numpy as np
import cv2

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the two left-most and two right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def measurePixelPerMatrix(pts):
	tl = pts[0]
	tr = pts[1]
	br = pts[2]
	bl = pts[3]

	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	if dA >= dB:
		pixelsPerMetric = dA / 6.3
	else:
		pixelsPerMetric = dB / 6.3

	return pixelsPerMetric

def findFocusZone(pts):
	xcordinates = []
	ycordinates = []

	for i in pts:
		current = i
		xcordinates.append(current[0])
		# print("current[0]: ", current[0])
		ycordinates.append(current[1])
		# print("current[1]: ", current[1])

	tlX = min(xcordinates)
	tlY = min(ycordinates)
	brX = max(xcordinates)
	brY = max(ycordinates)

	return np.array([tlX, tlY, brX, brY], dtype="int32")

