from __future__ import print_function
import numpy as np
import cv2
from imutils import perspective
from imutils import contours
import imutils
import utilities
from utilities import *
from tracker import *

pixelsPerMetric = None
refPts = []
count = 0
cropped = False
fps = None
framecount = 0

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videos/fps60_green_blue_5.MOV')
od = cv2.createBackgroundSubtractorMOG2()
tracker = EuclideanDistTracker()

def click_and_crop(event, x, y, flag, param):
    global refPts, cropped, count

    if event == cv2.EVENT_LBUTTONDOWN:
        curr = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        refPts.append((x,y))
        count += 1
        print(x,y, count)
        cv2.circle(image, (x,y), 3, (0, 255, 0), 2)
        cv2.imshow("image", image)

    if (len(refPts) == 4):
        cropped = True



_, image = cap.read()
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("r"):
		image = clone.copy()
		refPts.clear()
	elif key == ord("c"):
		break

if cropped is True:
	print("refpnts: ", refPts)
	refPts = np.array(refPts, dtype="int")
	refPts = order_points(refPts)
	print("refpnts: ", refPts)
	pixelsPerMetric = utilities.measurePixelPerMatrix(refPts)
	print("pixelsPerMetrics: ", pixelsPerMetric)


while True and pixelsPerMetric != None:
	ret, frame = cap.read()
	FocusPts = utilities.findFocusZone(refPts)
	tlX = FocusPts[0]
	tlY = FocusPts[1]
	brX = FocusPts[2]
	brY = FocusPts[3]
	roi = frame[tlY:brY, tlX:brX]

	if fps is None:
		fps = cap.get(cv2.CAP_PROP_FPS)
		print('frames per second =', fps)

	framecount += 1

	hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

	# # lower mask (0-10)
	# lower_red = np.array([0, 38, 38])
	# upper_red = np.array([20, 255, 255])
	# mask0 = cv2.inRange(hsv, lower_red, upper_red)
	#
	# # upper mask (170-180)
	# lower_red = np.array([165, 38, 38])
	# upper_red = np.array([180, 255, 255])
	# mask1 = cv2.inRange(hsv, lower_red, upper_red)

	# join my masks
	# mask = mask0 + mask1

	lower_blue = np.array([90, 25, 50])
	upper_blue = np.array([130, 255, 255])

	lower_green = np.array([35, 40, 50])
	upper_green = np.array([80, 255, 255])

	mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
	mask2 = cv2.inRange(hsv, lower_green, upper_green)

	myObject_blue = cv2.bitwise_and(roi, roi, mask=mask1)
	myObject_green = cv2.bitwise_and(roi, roi, mask=mask2)

	odmask_blue = od.apply(myObject_blue)
	odmask_green = od.apply(myObject_green)

	_, mask_blue = cv2.threshold(odmask_blue, 100, 255, cv2.THRESH_BINARY)
	_, mask_green = cv2.threshold(odmask_green, 100, 255, cv2.THRESH_BINARY)

	contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	finalContours_blue = []
	finalContours_green = []

	for c in contours_blue:
		# Calculate area and remove small elements
		area = cv2.contourArea(c)
		if area > 10:
			finalContours_blue.append([area, c])
			cv2.drawContours(myObject_blue, [c], -1, (0, 255, 0), 1)

	for c in contours_green:
		# Calculate area and remove small elements
		area = cv2.contourArea(c)
		if area > 10:
			finalContours_green.append([area, c])
			cv2.drawContours(myObject_green, [c], -1, (0, 255, 0), 1)

	finalContours_blue = sorted(finalContours_blue, key=lambda x: x[0], reverse=True)
	finalContours_green = sorted(finalContours_green, key=lambda x: x[0], reverse=True)
	detections_blue = []
	detections_green = []

	for cnt in finalContours_blue:
		x, y, w, h = cv2.boundingRect(cnt[1])
		detections_blue.append([x, y, w, h])

	for cnt in finalContours_green:
		x, y, w, h = cv2.boundingRect(cnt[1])
		detections_green.append([x, y, w, h])

	# object tracking
	boxes_ids_blue = tracker.update(detections_blue)
	boxes_ids_green = tracker.update(detections_green)

	distances_blue = []
	distances_green = []
	for box_id in boxes_ids_blue:
		x, y, w, h, id = box_id
		# cv2.putText(roi, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0,0), 2)
		cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 1)
		cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 1)

		cv2.circle(roi, (int(x), int(y)), 3, (255, 0, 0), -1)
		cv2.circle(roi, (int(x + w), int(y + h)), 3, (255, 0, 0), -1)
		cv2.circle(roi, (int(x + w), int(y)), 3, (255, 0, 0), -1)
		cv2.circle(roi, (int(x), int(y + h)), 3, (255, 0, 0), -1)

		tl = [x, y]
		tr = [x + w, y]
		br = [x + w, y + h]
		bl = [x, y + h]
		distances_blue.append([tl, tr, br, bl])
		# (tl, tr, br, bl) = box
		(tltrX, tltrY) = utilities.midpoint(tl, tr)
		(blbrX, blbrY) = utilities.midpoint(bl, br)

		(tlblX, tlblY) = utilities.midpoint(tl, bl)
		(trbrX, trbrY) = utilities.midpoint(tr, br)

		# compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric
	# draw the object sizes on the image0

	for box_id in boxes_ids_green:
		x, y, w, h, id = box_id
		# cv2.putText(roi, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0,0), 2)
		cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 1)
		cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 1)

		cv2.circle(roi, (int(x), int(y)), 3, (255, 0, 0), -1)
		cv2.circle(roi, (int(x + w), int(y + h)), 3, (255, 0, 0), -1)
		cv2.circle(roi, (int(x + w), int(y)), 3, (255, 0, 0), -1)
		cv2.circle(roi, (int(x), int(y + h)), 3, (255, 0, 0), -1)

		tl = [x, y]
		tr = [x + w, y]
		br = [x + w, y + h]
		bl = [x, y + h]
		distances_green.append([tl, tr, br, bl])
		# (tl, tr, br, bl) = box
		(tltrX, tltrY) = utilities.midpoint(tl, tr)
		(blbrX, blbrY) = utilities.midpoint(bl, br)

		(tlblX, tlblY) = utilities.midpoint(tl, bl)
		(trbrX, trbrY) = utilities.midpoint(tr, br)

		# compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric
	# draw the object sizes on the image0

	fingerLocation = []
	trackerLocation = []

	for distA in distances_green:
		for distB in distances_blue:
			minDist = 0
			shortestPath = []
			# if (distA != distB):
				# currtl = distA[0]
				# currtr = distA[1]
				# currbr = distA[2]
				# currbl = distA[3]
				#
				# nexttl = distB[0]
				# nexttr = distB[1]
				# nextbr = distB[2]
				# nextbl = distB[3]
			for i in range(0, 4):
				curr = distA[i]
				for j in range(0, 4):
					next = distB[j]
					dC = dist.euclidean(curr, next)
					dimC = dC / pixelsPerMetric
					if minDist == 0:
						minDist = dimC
						shortestPath = [curr, next]

					elif minDist > dimC:
						minDist = dimC
						shortestPath = [curr, next]

			list1 = [curr, framecount]
			# print(list1)
			fingerLocation.append(list1)
			trackerLocation = [next, framecount]
			print("trackerLocation: ", trackerLocation)

			for loc1 in fingerLocation:
				print("loc1: ", loc1)






			# print("shortestpath[0]: ", shortestPath[0])
			# print("shortestpath[1]: ", shortestPath[1])
			cv2.putText(roi, "{:.1f}in".format(minDist), shortestPath[0], cv2.FONT_HERSHEY_SIMPLEX,
						0.65, (0, 0, 255), 2)
			cv2.line(roi, shortestPath[0], shortestPath[1], (0, 0, 255), 1)

	cv2.imshow('frame_green', np.hstack([roi, myObject_green]))
	cv2.imshow('frame_blue', np.hstack([roi, myObject_blue]))

	# if cv2.waitKey(1) == ord('q'):
	# 	break

	# This is for playing the video with a key press
	key = cv2.waitKey(0)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()