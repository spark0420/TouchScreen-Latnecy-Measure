from __future__ import print_function
from imutils import perspective
from imutils import contours
import imutils
import cv2
import utilities
from utilities import *
import tracker
from tracker import *
import numpy as np

def findFinalContours(contours):

    finalContours = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        finalContours.append([area, c])

    finalContours = sorted(finalContours, key=lambda x: x[0], reverse=True)

    return finalContours
    #
    # for cnt in finalContours:
    #     x,y,w,h = cv2.boundingRect(cnt[1])
    #     detections.append([x,y,w,h])
    #
    # #object tracking
    # boxes_ids = tracker.update(detections)
    # distances = []
    #
    # for box_id in boxes_ids:
    #     x,y,w,h,id = box_id
    #     cv2.putText(roi, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0,0), 2)
    #     cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #     cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #
    #     cv2.circle(roi, (int(x), int(y)), 3, (255, 0, 0), -1)
    #     cv2.circle(roi, (int(x + w), int(y + h)), 3, (255, 0, 0), -1)
    #     cv2.circle(roi, (int(x + w), int(y)), 3, (255, 0, 0), -1)
    #     cv2.circle(roi, (int(x), int(y + h)), 3, (255, 0, 0), -1)
    #
    #     tl = [x,y]
    #     tr = [x+w, y]
    #     br = [x+w, y+h]
    #     bl = [x, y+h]
    #     distances.append([tl, tr, br, bl])
    #     # (tl, tr, br, bl) = box
    #     (tltrX, tltrY) = utilities.midpoint(tl, tr)
    #     (blbrX, blbrY) = utilities.midpoint(bl, br)
    #
    #     (tlblX, tlblY) = utilities.midpoint(tl, bl)
    #     (trbrX, trbrY) = utilities.midpoint(tr, br)
    #
    #     cv2.circle(roi, (int(tltrX), int(tltrY)), 3, (255, 0, 0), -1)
    #     cv2.circle(roi, (int(blbrX), int(blbrY)), 3, (255, 0, 0), -1)
    #     cv2.circle(roi, (int(tlblX), int(tlblY)), 3, (255, 0, 0), -1)
    #     cv2.circle(roi, (int(trbrX), int(trbrY)), 3, (255, 0, 0), -1)
    #
    #     cv2.line(roi, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 1)
    #     cv2.line(roi, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 1)
    #
    #     # compute the Euclidean distance between the midpoints
    #     dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    #     dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    #     # if the pixels per metric has not been initialized, then
    #     # compute it as the ratio of pixels to supplied metric
    #
    #     # (in this case, inches)
    #     # if pixelsPerMetric is None:
    #     #     pixelsPerMetric = dB / 6.3
    #     #     print("pixelsPerMetrics: ", pixelsPerMetric)
    #
    #     dimA = dA / pixelsPerMetric
    #     dimB = dB / pixelsPerMetric
    #     # draw the object sizes on the image0
    #     cv2.putText(roi, "{:.1f}in".format(dimA),
    #                 (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.65, (255, 255, 255), 2)
    #     cv2.putText(roi, "{:.1f}in".format(dimB),
    #                 (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.65, (255, 255, 255), 2)
    #
    # # distances.append([tl, tr, br, bl])
    # for distA in distances:
    #     for distB in distances:
    #         minDist = 0
    #         shortestPath = []
    #         if (distA != distB):
    #             # currtl = distA[0]
    #             # currtr = distA[1]
    #             # currbr = distA[2]
    #             # currbl = distA[3]
    #             #
    #             # nexttl = distB[0]
    #             # nexttr = distB[1]
    #             # nextbr = distB[2]
    #             # nextbl = distB[3]
    #             for i in range(0,4):
    #                 curr = distA[i]
    #                 for j in range(0,4):
    #                     next = distB[j]
    #                     dC = dist.euclidean(curr, next)
    #                     dimC = dC / pixelsPerMetric
    #                     if minDist == 0:
    #                         minDist = dimC
    #                         shortestPath = [curr, next]
    #                     elif minDist > dimC:
    #                         minDist = dimC
    #                         shortestPath = [curr, next]
    #
    #             # print("shortestpath[0]: ", shortestPath[0])
    #             # print("shortestpath[1]: ", shortestPath[1])
    #             cv2.putText(roi, "{:.1f}in".format(minDist),shortestPath[0], cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.65, (0, 0, 255), 2)
    #             cv2.line(roi, shortestPath[0], shortestPath[1], (0, 255, 0), 1)
    #
