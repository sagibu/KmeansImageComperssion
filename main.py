import cv2
from imageComperssion import compress


def main():
    compressed = compress('panda.jpg')
    cv2.imwrite("newPanda.jpg", compressed)


if __name__ == "__main__":
    main()