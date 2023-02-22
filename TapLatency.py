import numpy as np
import cv2
import utilities
from utilities import *
import itertools

refPts = []
fps = None
framecount = 0
count = 0
inputList = []
outputList = []
currInput = 0
currOutput = 0
inputCount = 0
outputCount = 0
detected = False

cap = cv2.VideoCapture('videos/tapButton4.MOV')

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
    roi2 = roi.copy()

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('frames per second =', round(fps))

    framecount += 1
    cv2.putText(userScreen, "fps: " + str(round(fps)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 25, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    _, mask_blue = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    listLen = len(contours_blue)

    if currInput != 0:
        cv2.putText(userScreen, "touch detected at frame# " + str(currInput), (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 2)


    if listLen == 0:
        detected = False
        print("ListLen: 0")
    elif listLen > 0 and detected == False:
        # print("blue detected at frame# ", framecount)
        currOutput = framecount
        outputCount += 1
        outputList.append(currOutput)
        print("Screen output: ", framecount)
        cv2.putText(userScreen, "blue detected at frame# " + str(currOutput), (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 2)
        detected = True
    else:
        cv2.putText(userScreen, "blue detected at frame# " + str(currOutput), (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 2)


    cv2.imshow('userScreen', userScreen)
    cv2.imshow('mask', mask)
    # cv2.imshow('roi', roi2)

    # if cv2.waitKey(1) == ord('q'):
    # 	break

    # This is for playing the video with a key press
    key = cv2.waitKey(0)
    if key == ord('i'):
        currInput = framecount
        inputCount += 1

        if inputCount <= outputCount:
            outputList.pop()
            print("input click was later than output. Deleting the current input and most recent output data")
        else:
            inputList.append(currInput)
            print("User input: ", currInput)
    elif key == 27:
        break

for (i, j) in zip(inputList, outputList):
    latency = (j - i) / round(fps)
    latency = round(latency, 3)
    print("input: ", i, " output: ", j, " latency: ", latency, "sec")


cap.release()
cv2.destroyAllWindows()



