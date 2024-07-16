import cv2
import numpy as np
import ModifiedHandTrackingModule as htm
import time
import pyautogui

wCam, hCam = 640, 480
frameR = 100 # frame reduction
smoothening = 5
pLocX, pLogY = 0,0
cLocX, pLocY = 0,0
pTime = 0



cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)

wScr, hScr = pyautogui.size()

while True:
    success, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)
    lmlist, bbox = detector.findPosition(frame)

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        fingers = detector.fingersUp()
        # for the frame reduction
        cv2.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 0), 2)

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            cLocX = pLocX + (x3 - pLocX) / smoothening
            cLocY = pLocY + (y3 - pLocY) / smoothening

            pyautogui.moveTo(wScr-x3, y3)
            cv2.circle(frame, (x1, y1), radius=10, color=(255, 0, 0), thickness=cv2.FILLED)
            pLocX, pLogY = cLocX, cLocY

        if fingers[1] == 1 and fingers[2] == 1:
            length, frame, lineInfo = detector.findDistance(8,12, frame)
            # print(length)
            if length < 40:
                cv2.circle(frame, (lineInfo[4], lineInfo[5]), radius=10, color=(0, 255, 0), thickness=cv2.FILLED)
                pyautogui.click()

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'{int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN,
                3, (250, 0, 0), 3)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
