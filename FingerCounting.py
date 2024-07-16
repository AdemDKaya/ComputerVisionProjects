import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "images"
myList = os.listdir(folderPath)
print(myList)
overLayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overLayList.append(image)
print(len(overLayList))
pTime = 0

detector = htm.HandDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHand(img)
    lmList = detector.find_position(img, draw=False)

    if len(lmList) != 0:
        fingers= []

        # right thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # for the left thumb (we only change less than to grater than)
        # if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
        #     fingers.append(1)
        # else:
        #     fingers.append(0)

        # 4 right fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, _ = overLayList[totalFingers - 1].shape
        img[0:h, 0:w] = overLayList[totalFingers - 1]

        cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (250, 0, 0), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)
