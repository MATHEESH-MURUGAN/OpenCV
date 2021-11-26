import cv2
import os
from mediapipe import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import MouseAI as mai
import time
import autopy

##########################
wCam=640
hCam=480
framered=100
smoothening=7
#########################

pTime=0
pX, pY=0, 0
cX, cY=0, 0
cap = cv2.VideoCapture(0)
detector = mai.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1, y1, x2, y2)

    # 3. Check which fingers are up
    fingers = detector.fingersUp()
    print(fingers)
    cv2.rectangle(img, (framered, framered), (wCam - framered, hCam - framered),
                  (255, 0, 255), 2)
    # 4. Only Index Finger : Moving Mode
    if fingers[1] == 1 and fingers[2] == 0:
        # 5. Convert Coordinates
        x3 = np.interp(x1, (framered, wCam - framered), (0, wScr))
        y3 = np.interp(y1, (framered, hCam - framered), (0, hScr))
        # 6. Smoothen Values
        cX = pX + (x3 - pX) / smoothening
        cY = pY + (y3 - pY) / smoothening

        # 7. Move Mouse
        autopy.mouse.move(wScr - cX, cY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        pX, pY = cX, cY

    # 8. Both Index and middle fingers are up : Clicking Mode
    if fingers[1] == 1 and fingers[2] == 1:
        # 9. Find distance between fingers
        length, img, lineInfo = detector.findDistance(8, 12, img)
        print(length)
        # 10. Click mouse if distance short
        if length <40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    (255, 0, 0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break








