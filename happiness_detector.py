# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 23:51:23 2020

@author: sinha
"""

import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detectFaces(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        gray_roi = gray[y:y+h, x:x+w]
        color_roi = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(gray_roi, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(color_roi, (ex, ey), (ex+ew, ey+eh), (0,0,255), 2)
        
        smiles = smile_cascade.detectMultiScale(gray_roi, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(color_roi, (sx, sy), (sx+sw, sy+sh), (100, 200, 255), 2)
    
    return frame
    
video_capture = cv2.VideoCapture(0)
while True:
    new, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detectedFaces = detectFaces(gray, frame)
    cv2.imshow('Video', detectedFaces)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
