from typing import List

import numpy as np
from random import *

MAX_DISTANCE = 100000


def randomize_centroids(k, data):
    centroids = dict()
    for i in range(k):
        centroids[i] = data[randint(0, data.shape[0] - 1)]
    return centroids


def cluster(centroids, data):
    clusters = dict()
    for i in range(data.shape[0]):
        min_distance = MAX_DISTANCE
        for key, value in centroids.items():
            dist = np.linalg.norm(data[i] - value)
            if dist < min_distance:
                min_distance = dist
                clusters[i] = key
    return clusters


def calculate_centroids(k, clusters, data):
    centroids = dict()
    clustersNumOfItems = dict()
    totForCluster = dict()
    for i in range(k):
        totForCluster[i] = [0, 0, 0]
        clustersNumOfItems[i] = 0

    for key, value in clusters.items():
        totForCluster[value] += data[key]
        clustersNumOfItems[value] += 1

    for key, value in totForCluster.items():
        if clustersNumOfItems[key] != 0:
            centroids[key] = value / clustersNumOfItems[key]
    return centroids


def kmeans(k, data, n_iters=10, tolerance=5):
    centroids = randomize_centroids(k, data)
    for i in range(n_iters):
        clusters = cluster(centroids, data)
        new_centroids = calculate_centroids(k, clusters, data)
        if np.linalg.norm(np.subtract(list(centroids.values()), list(new_centroids.values()))) <= tolerance:
            return new_centroids, clusters
        centroids = new_centroids

    return centroids, clusters
