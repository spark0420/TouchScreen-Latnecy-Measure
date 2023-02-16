from __future__ import print_function
from imutils import perspective
from imutils import contours
import imutils
import cv2
import utilities
from utilities import *
import numpy as np
from tracker import *

pixelsPerMetric = None
refPts = []
count = 0
cropped = False
fps = None
framecount = 0

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videos/fps60_blue.MOV')
od = cv2.createBackgroundSubtractorMOG2()
tracker = EuclideanDistTracker()

# lower_blue = np.array([110, 50, 50])
# upper_blue = np.array([130, 255, 255])

lower_blue = np.array([90, 25, 50])
upper_blue = np.array([130, 255, 255])

lower_green = np.array([35, 30, 50])
upper_green = np.array([80, 255, 255])

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

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('frames per second =',fps)

    framecount += 1
    print(framecount)

    FocusPts = utilities.findFocusZone(refPts)
    tlX = FocusPts[0]
    tlY = FocusPts[1]
    brX = FocusPts[2]
    brY = FocusPts[3]
    roi = frame[tlY:brY, tlX:brX]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask = cv2.inRange(hsv, lower_green, upper_green)
    myObject = cv2.bitwise_and(roi, roi, mask=mask)
    odmask = od.apply(myObject)

    _, mask = cv2.threshold(odmask, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []

    for c in contours:
        #Calculate area and remove small elements
        area = cv2.contourArea(c)
        if area > 10:
            finalContours.append([area, c])
            cv2.drawContours(myObject, [c], -1, (0, 255, 0), 1)

    finalContours = sorted(finalContours, key=lambda x: x[0], reverse=True)
    detections = []

    for cnt in finalContours:
        x,y,w,h = cv2.boundingRect(cnt[1])
        detections.append([x,y,w,h])

    # object tracking
    boxes_ids = tracker.update(detections)
    distances = []

    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        # cv2.putText(roi, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0,0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 1)

        cv2.circle(roi, (int(x), int(y)), 3, (255, 0, 0), -1)
        cv2.circle(roi, (int(x + w), int(y + h)), 3, (255, 0, 0), -1)
        cv2.circle(roi, (int(x + w), int(y)), 3, (255, 0, 0), -1)
        cv2.circle(roi, (int(x), int(y + h)), 3, (255, 0, 0), -1)

        tl = [x,y]
        tr = [x+w, y]
        br = [x+w, y+h]
        bl = [x, y+h]
        distances.append([tl, tr, br, bl])
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

    for distA in distances:
        for distB in distances:
            minDist = 0
            shortestPath = []
            if (distA != distB):
                # currtl = distA[0]
                # currtr = distA[1]
                # currbr = distA[2]
                # currbl = distA[3]
                #
                # nexttl = distB[0]
                # nexttr = distB[1]
                # nextbr = distB[2]
                # nextbl = distB[3]
                for i in range(0,4):
                    curr = distA[i]
                    for j in range(0,4):
                        next = distB[j]
                        dC = dist.euclidean(curr, next)
                        dimC = dC / pixelsPerMetric
                        if minDist == 0:
                            minDist = dimC
                            shortestPath = [curr, next]
                        elif minDist > dimC:
                            minDist = dimC
                            shortestPath = [curr, next]

                # print("shortestpath[0]: ", shortestPath[0])
                # print("shortestpath[1]: ", shortestPath[1])
                cv2.putText(roi, "{:.1f}in".format(minDist),shortestPath[0], cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 255), 2)
                cv2.line(roi, shortestPath[0], shortestPath[1], (0, 255, 0), 1)

    cv2.imshow('odmask', odmask)
    cv2.imshow('frame', np.hstack([roi, myObject]))

    # This is for auto-play
    # if cv2.waitKey(1) == ord('q'):
    #     break

    # This is for playing the video with a key press
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()