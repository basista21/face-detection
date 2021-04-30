# Usage: python face-detect-dnn.py [optional.jpg]

import numpy as np
import sys
import cv2
from imutils.video import VideoStream
import imutils
import time

# Paths to prototxt file and Caffe model
prototxtPath = "deploy.prototxt.txt"
caffemodelPath = "res10_300x300_ssd_iter_140000.caffemodel"

conf = 0.30                          # Confidence level threshold
thickness = 2                       # Thickness of rectangle 
blue = (247, 173, 62)               # Color in BGR
white = (255, 255, 255)             # Color in BGR
font = cv2.FONT_HERSHEY_SIMPLEX     # Font style
meanValues = (104.0, 177.0, 124.0)  # RGB mean values from ImageNet training set

# Load model
net = cv2.dnn.readNetFromCaffe(prototxtPath, caffemodelPath)

def drawRectangle(image, color, t):
    (x, y, x1, y1) = t
    h = y1 - y
    w = x1 - x
    barLength = int(h / 8)
    cv2.rectangle(image, (x, y-barLength), (x+w, y), color, -1)
    cv2.rectangle(image, (x, y-barLength), (x+w, y), color, thickness)
    cv2.rectangle(image, (x, y), (x1, y1), color, thickness)
    return image

# Changes font scale as a function of face box size
def changeFontScale(h, fontScale):
    baseHeight = 108        # Height of model image
    fontScale = h/108 * fontScale
    return fontScale

def detectFaces(image):
    h, w, _ = image.shape
    resizedImage = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(resizedImage, 1.0, (300, 300), meanValues)

    net.setInput(blob)
    faces = net.forward()

    for i in range(0, faces.shape[2]):
        confidence = faces[0, 0, i, 2]

        if confidence > conf:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            fontScale = changeFontScale(y1-y, 0.4)
            image = drawRectangle(image, blue, (x, y, x1, y1))

            # Display confidence level in %
            text = "{:0.2f}%".format(confidence * 100)
            textY = y - 2
            if (textY - 2 < 20): textY = y + 20 
            cv2.putText(image, text, (x, textY), font, fontScale, white, 1)

    return image

def useWebcam():
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        frame = detectFaces(frame)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) > 0:
            break

    cv2.destroyAllWindows()
    vs.stop()

def useImage():
    image = cv2.imread(sys.argv[1])     # Read image
    image = detectFaces(image)
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)

def main():
    if len(sys.argv) == 1:
        useWebcam()
    elif len(sys.argv) == 2:
        useImage()
    else:
        print("Usage: python face-detect-dnn.py [optional.jpg]")
        exit()

if __name__ == "__main__":
    main()