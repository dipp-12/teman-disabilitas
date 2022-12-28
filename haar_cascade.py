# -*- coding: utf-8 -*-
"""Haar Cascade.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14Uyd_Veyap49vAcj_tw57DkaTQcXgDUt
"""

import numpy as np
import cv2
f_cascade = cv2.CascadeClassifier("face.xml")
e_cascade = cv2.CascadeClassifier("eye.xml")

image = cv2.imread("actor.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = f_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = e_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()