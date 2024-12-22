# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:41:40 2023

@author: sara
"""

import cv2 as cv

face_cascade = cv.CascadeClassifier("C:/Users/sara/Desktop/haarcascade_frontalface_default.xml")

img = cv.imread("C:/Users/sara/Desktop/sara.jpg")#read the input image

faces = face_cascade.detectMultiScale(img, 1.1, 4)#detect faces

#Draw rectangle around the faces
for(x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    
#Export the result
cv.imwrite("C:/Users/sara/Desktop/face_detected.png", img) 
print("photo successfully exported")   
