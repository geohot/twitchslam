#!/usr/bin/env python3
import time
import cv2
from display import Display

W = 1920//2
H = 1080//2

disp = Display(W, H)

def process_frame(img):
  img = cv2.resize(img, (W,H))
  disp.paint(img)

if __name__ == "__main__":
  cap = cv2.VideoCapture("test.mp4")

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      break


