# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 19:18:00 2018

@author: 58011256
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

img = cv2.imread('C:/Users/58011256/Desktop/COde/Python/image segmentation/test.jpg')

reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(reshaped)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 10), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
