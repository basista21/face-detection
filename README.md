# Face Detection
Face detection using the frontal face Haar Cascade in OpenCV

## Haar Cascade Classifiers
Haar Cascade files can be downloaded from the [OpenCV repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

## Usage
Usage: **python faceDetect.py [optional.jpg]**

If only one command line argument is supplied (the program itself -- *faceDetect.py*), video stream from the webcam will be used as input.
If there are exactly two command line arguments and an image is provided as the second command line argument, the image will be used as input.

## Example
Running **python faceDetect.py dominic.jpg** we get:

![Image of Dominic with Face Detection](https://github.com/basista21/faceDetection/blob/main/dominic_out.jpg)
