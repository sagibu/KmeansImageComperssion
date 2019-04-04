import cv2
from kmeans import *

RGB_SIZE = 3

def compress(image_path, n_color=2, n_iterations=10, n_images=3, err_tol=100):
    image = cv2.imread(image_path)
    height = image.shape[1]
    width = image.shape[0]
    image = image.reshape(width * height, RGB_SIZE)

    calculated_image = np.ndarray(image.shape)

    (centroids, clusters) = kmeans(n_color, image, n_iters=int(n_iterations))
    for key, value in clusters.items():
        calculated_image[key] = centroids[value]
    return calculated_image.reshape(width, height, RGB_SIZE)
