import numpy as np
import cv2
from random import *
import time


def init(k):
    centroids = dict()
    for i in range(k):
        centroids[i] = image[randint(0, (int)(image.size / 3) - 1)]
        print(centroids[i])
    return centroids


def kmeans(k, centroids, panda):
    clusters = dict()
    for i in range(panda.shape[0]):
        min = 1000000 #bigger than max possible distance
        for key, value in centroids.items():
            dist = np.linalg.norm(panda[i] - value)
            if dist < min:
                min = dist
                clusters[i] = key

    clustersNumOfItems = dict()
    totForCluster = dict()
    for k, v in centroids.items():
        totForCluster[k] = [0, 0, 0]
        clustersNumOfItems[k] = 0

    for key, value in clusters.items():
        totForCluster[value] += panda[key]
        clustersNumOfItems[value] += 1

    print(clustersNumOfItems)
    for key, value in totForCluster.items():
        if clustersNumOfItems[key] != 0:
            centroids[key] = value / clustersNumOfItems[key]

    return (centroids, clusters)


#
k = 2
image = cv2.imread('panda.jpg')
height = image.shape[1]
width = image.shape[0]
image = image.reshape(width * height, 3)

calculatedImage = np.ndarray(image.shape)

for j in range(4):
    start = time.time()
    (centroids, clusters) = kmeans(k, init(k), image)
    end = time.time()
    print("time init: ", end - start)
    for i in range(10):
        print("iterate num " + str(i))
        (centroids, clusters) = kmeans(k, centroids, image)
        for key, value in clusters.items():
            calculatedImage[key] = centroids[value]
        calculatedImage = calculatedImage.reshape(width, height, 3)
        cv2.imwrite("panda" + str(i) + "seed" + str(j) + ".jpg", calculatedImage)
        calculatedImage = calculatedImage.reshape(width * height, 3)

# kmeansRes = KMeans(n_clusters=2, random_state=0).fit(image)
# kmeansRes.cluster_centers_
