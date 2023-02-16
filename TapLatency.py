import numpy as np
import cv2
import utilities
from utilities import *

refPts = []
fps = None
framecount = 0
count = 0
input = 0
output = 0

cap = cv2.VideoCapture('videos/tap1.MOV')

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
    # print("refpnts: ", refPts)
    refPts = np.array(refPts, dtype="int")
    refPts = order_points(refPts)
    # print("refpnts: ", refPts)


while True and cropped is True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    userScreen = frame.copy()
    FocusPts = utilities.findFocusZone(refPts)
    tlX = FocusPts[0]
    tlY = FocusPts[1]
    brX = FocusPts[2]
    brY = FocusPts[3]
    roi = frame[tlY:brY, tlX:brX]

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('frames per second =', round(fps))

    framecount += 1
    cv2.putText(userScreen, "fps: " + str(fps) + str(input), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 25, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    _, mask_blue = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    listLen = len(contours_blue)

    if input != 0:
        cv2.putText(userScreen, "touch detected at frame# " + str(input), (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 2)
    if listLen > 0:
        # print("blue detected at frame# ", framecount)
        if output == 0:
            output = framecount
            print("Screen output: ", framecount)
            cv2.putText(userScreen, "blue detected at frame# " + str(output), (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            cv2.putText(userScreen, "blue detected at frame# " + str(output), (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


    # cv2.imshow('roiandmask', np.hstack([roi, mask]))
    cv2.imshow('userScreen', userScreen)
    cv2.imshow('mask', mask)

    # if cv2.waitKey(1) == ord('q'):
    # 	break

    # This is for playing the video with a key press
    key = cv2.waitKey(0)
    if key == ord('i'):
        input = framecount
        print("User input: ", input)
    elif key == 27:
        break

latency = (output -input) / round(fps)
latency = round(latency, 3)
print("latency: ", latency, "sec")


cap.release()
cv2.destroyAllWindows()



