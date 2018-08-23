# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 21:08:29 2018

@author: 58011256
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    ORIG = None
    
    def __init__(self, image, clusters):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self):
    
        #read image
        img = cv2.imread(self.IMAGE)
        self.ORIG = img.copy()
        
        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS, init = 'k-means++', random_state = 42)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)
    
    def show(self):
        cv2.imshow('img',self.ORIG)
        

img = 'C:/Users/58011256/Desktop/COde/Python/image segmentation/test.jpg'
clusters = 3
dc = DominantColors(img, clusters) 
img = cv2.imread('C:/Users/58011256/Desktop/COde/Python/image segmentation/test1.jpg')
#cv2.imshow('image',img)
colors = dc.dominantColors()
#cv2.imshow('Test',dc.mark_label(212,181,213))

print(colors)



'''
#get rgb values from image to 1D array
r, g, b = cv2.split(img)
r = r.flatten()
g = g.flatten()
b = b.flatten()

#plotting 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(r, g, b)
plt.show()
'''

'''
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(reshaped)

# Visualising the clusters
plt.scatter(reshaped[y_kmeans == 0, 0], reshaped[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(reshaped[y_kmeans == 1, 0], reshaped[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
'''