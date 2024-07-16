import tkinter as tk
from tkinter import Button, Label
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image, ImageTk

genai.configure(api_key="YOUR_API_KEY_HERE")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

class HandTrackingApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_start = Button(window, text="Start Hand Tracking", command=self.start_tracking)
        self.btn_start.pack()

        self.btn_stop = Button(window, text="Stop Hand Tracking", command=self.stop_tracking)
        self.btn_stop.pack()

        self.output_text_area = Label(window, text="", wraplength=300)
        self.output_text_area.pack()

        self.update_flag = False

        self.prev_pos = None
        self.canvas_img = None
        self.image_combined = None

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def getHandInfo(self, img):
        hands, img = detector.findHands(img, draw=False, flipType=True)
        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = detector.fingersUp(hand)
            return fingers, lmList
        else:
            return None

    def draw(self, info, prev_pos, canvas_img):
        fingers, lmList = info
        current_pos = None
        if fingers == [0, 1, 0, 0, 0]:
            current_pos = lmList[8][0:2]
            if prev_pos is None:
                prev_pos = current_pos
            cv2.line(canvas_img, current_pos, prev_pos, (255, 0, 255), 5)
        elif fingers == [1, 0, 0, 0, 0]:
            canvas_img = np.zeros_like(canvas_img)
        return current_pos, canvas_img

    def sendToAI(self, model, canvas_img, fingers):
        if fingers == [1, 1, 1, 1, 0]:
            pil_image = Image.fromarray(canvas_img)
            response = model.generate_content(["Solve this math problem", pil_image])
            return response.text

    def start_tracking(self):
        self.update_flag = True
        self.update()

    def stop_tracking(self):
        self.update_flag = False

    def update(self):
        if self.update_flag:
            success, img = cap.read()
            img = cv2.flip(img, 1)

            if self.canvas_img is None:
                self.canvas_img = np.zeros_like(img)

            info = self.getHandInfo(img)
            if info:
                fingers, lmList = info
                self.prev_pos, self.canvas_img = self.draw(info, self.prev_pos, self.canvas_img)
                output_text = self.sendToAI(model, self.canvas_img, fingers)
                if output_text:
                    self.output_text_area.config(text=output_text)

            self.image_combined = cv2.addWeighted(img, 0.7, self.canvas_img, 0.3, 0)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.image_combined, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(10, self.update)

    def on_closing(self):
        self.stop_tracking()
        cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackingApp(root, "Hand Tracking GUI")
