# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:20:15 2018

@author: 58011256
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/58011256/Desktop/COde/Python/image segmentation/test1.jpg')



kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
'''
equ = cv2.equalizeHist(closing)
res = np.hstack((closing,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res)
'''
blur = cv2.GaussianBlur(closing,(5,5),0)
edges = cv2.Canny(blur,50,200)

'''
plt.subplot(121),plt.imshow(closing,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
'''
plt.subplot(121),plt.imshow(blur,cmap = 'gray')
plt.title('Closing+Blur Image'), plt.xticks([]), plt.yticks([])

cv2.imwrite("cleanIMG.jpg",blur)

plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()