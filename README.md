# Face Detection
Face detection using Haar Cascades (frontal face and profile face) and Deep Neural Network (DNN) module in OpenCV.

## Haar Cascade Classifiers
Haar Cascade files can be downloaded from the [OpenCV repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

## Deep Neural Network
The file for the pre-trained Caffe model can be found [here](https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel).

## Usage
Usage (using haar cascades): **python face-detect-haar.py [optional.jpg]**<br>
Usage (using dnn)          : **python face-detect-dnn.py [optional.jpg]**

If only one command line argument is supplied (the program itself e.g. *face-detect-haar.py*), video stream from the webcam will be used as input.
If there are exactly two command line arguments and an image is provided as the second command line argument, the image will be used as input.

## Examples
Running **python face-detect-haar.py dominic.jpg**:

![Image of Dominic with Face Detection Using Haar Cascades](https://github.com/basista21/faceDetection/blob/main/dominic_out.jpg)

Running **python face-detect-dnn.py dominic.jpg**:

![Image of Dominic with Face Detection Using DNN](https://user-images.githubusercontent.com/74373754/116709904-bc8e5380-aa03-11eb-83c9-32d9ea33fed5.png)

*The DNN detector successfully detected the third face, but with only 30.02% confidence.*

### Comparison

Haar cascades vs DNN:

![image](https://user-images.githubusercontent.com/74373754/116710833-c6648680-aa04-11eb-95c3-dbaa46b570f9.png)

*A false positive and a false negative for the haar face detector. DNN successfully detected the face.*

<br><br>

![image](https://user-images.githubusercontent.com/74373754/116711745-bd27e980-aa05-11eb-9b4d-5ff31fe8bdeb.png)

*A true positive and a false positive for each.*









