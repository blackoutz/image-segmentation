# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 19:15:38 2018

@author: 58011256
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

img = cv2.imread('C:/Users/58011256/Desktop/COde/Python/image segmentation/test.jpg')

#cv2.imshow('ORIG',img)
# Convert BGR to HSV
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#cv2.imshow('frame',img)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#plt.imshow('hsv',hsv)
# define range of blue color in HSV
lower_blue = np.array([137,42,144])
upper_blue = np.array([207,92,255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(img,img, mask= mask)
cv2.imshow('frame',img)
#cv2.imshow('mask',mask)
cv2.imshow('res',res)
