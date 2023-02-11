# import the necessary packages
from __future__ import print_function
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
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

# load our input image, convert it to grayscale, and blur it slightly
path = 'photos/nocase5.JPG'
cap = cv2.VideoCapture('videos/IMG_2246.mov')
# cap = cv2.VideoCapture('videos/IMG_2243-1.mov')

while True:
    _, image = cap.read()
    # image = cv2.imread(path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray,200, 200)
    edged = cv2.dilate(edged, None, iterations=5)
    edged = cv2.erode(edged, None, iterations=3)


    # cv2.imshow("edged", edged)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    pixelsPerMetric = None
    # sort the contours from left-to-right and initialize the bounding box
    # point colors
    # (cnts, _) = contours.sort_contours(cnts)
    finalContours = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 100:
            continue

        finalContours.append([area, c])

    finalContours = sorted(finalContours, key = lambda x:x[0], reverse=True)

    colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

    # loop over the contours individually
    for c in finalContours:
        # if the contour is not sufficiently large, ignore it
        # area = cv2.contourArea(c)
        # if area < 150:
        # 	continue
        # compute the rotated bounding box of the contour, then
        # draw the contours
        box = cv2.minAreaRect(c[1])
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        # show the original coordinates
        # print("Object #{}:".format(i + 1))
        # print(box)

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = order_points(box)
        print("box: ", box)
        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 1)


        # show the re-ordered coordinates
        # print(rect.astype("int"))
        # print("")

        # loop over the original points and draw them
        # for ((x, y), color) in zip(rect, colors):
        # 	cv2.circle(image, (int(x), int(y)), 5, color, -1)
        for(x, y) in box:
            cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
        # draw the object num at the top-left corner
        # cv2.putText(image, "Object #{}".format(i + 1),
        # 			(int(rect[0][0] - 15), int(rect[0][1] - 15)),
        # 			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.circle(image, (int(tltrX), int(tltrY)), 3, (255, 0,0), -1)
        cv2.circle(image, (int(blbrX), int(blbrY)), 3, (255, 0, 0), -1)
        cv2.circle(image, (int(tlblX), int(tlblY)), 3, (255, 0, 0), -1)
        cv2.circle(image, (int(trbrX), int(trbrY)), 3, (255, 0, 0), -1)

        cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 1)
        cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 1)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 6.3

        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
        # draw the object sizes on the image0
        cv2.putText(image, "{:.1f}in".format(dimA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
        cv2.putText(image, "{:.1f}in".format(dimB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)

    cv2.imshow("Image", image)
    # cv2.waitKey(0)

    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()