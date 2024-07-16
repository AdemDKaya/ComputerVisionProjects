import cv2
import ModifiedHandTrackingModule as htm
from time import sleep
from pynput.keyboard import Controller

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.handDetector(detectionCon=0.8)

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

finalText = ""

keyboard = Controller()
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.position
        w, h = button.size
        cv2.rectangle(img, button.position, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 10, y + 30), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), 3)
    return img

class Button():
    def __init__(self, position, text, size=[40, 40]):
        self.position = position
        self.text = text
        self.size = size


buttonList = []

for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([j * 50 + 50, 50 * i + 50], key))
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img)
    img = drawAll(img, buttonList)

    if lmList:
        for button in buttonList:
            x, y = button.position
            w, h = button.size
            if x < lmList[8][1] < x + w and y < lmList[8][2] < y + h:
                cv2.rectangle(img, (x-5, y-5), (x + w+5, y + h+5), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 10, y + 30), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 255, 255), 3)

                length, _, _ = detector.findDistance(8, 12, img, draw=False)
                # print(length)
                if length < 25:
                    keyboard.press(button.text)
                    cv2.rectangle(img, button.position, (x + w, y + h), (0, 250, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 10, y + 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 255, 255), 3)

                    finalText += button.text
                    sleep(0.30)


    cv2.rectangle(img, (50, 350), (590, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 425), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 255, 255), 3)

    cv2.imshow("Virtual Keyboard", img)
    cv2.waitKey(1)
