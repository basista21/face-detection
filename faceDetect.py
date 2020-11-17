import cv2
import sys

# Load the cascade classifiers
frontalFaceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Color in BGR
blue = (247, 173, 62)

# Thickness of rectangle drawn around faces
thickness = 2

def detectFace(grayscale, image):
	# Detects faces in the image using the face cascade
	faces = frontalFaceCascade.detectMultiScale(
		grayscale,
		scaleFactor=1.05,
		minNeighbors=5,
		minSize=(30,30),
	)

	# Draw a rectangle around each detected face
	for (x, y, w, h) in faces:
		barLength = int(h / 8)
		barWidth = w
		cv2.rectangle(image, (x, y-barLength), (x+barWidth, y), blue, -1)
		cv2.rectangle(image, (x, y-barLength), (x+barWidth, y), blue, thickness)
		cv2.rectangle(image, (x, y), (x+w, y+h), blue, thickness)

def useWebcam():
	video  = cv2.VideoCapture(0)
	while True:
		_, frame = video.read()

		# Convert frame to grayscale
		grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		detectFace(grayscale, frame)

		# Flip the frame
		frame = cv2.flip(frame, 1)

		cv2.imshow("Face Detection", frame)
		if cv2.waitKey(1) > 0:
			break

	video.release()
	cv2.destroyAllWindows()

def useImage():
	# Read the image
	image = cv2.imread(sys.argv[1])

	# Convert image to grayscale
	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	detectFace(grayscale, image)

	cv2.imshow("Face Detection", image)
	cv2.waitKey(0)

def main():
	if len(sys.argv) == 1:
		useWebcam()
	elif len(sys.argv) == 2:
		useImage()
	else:
		print("Usage: python faceDetect.py [optional.jpeg]")
		exit()

if __name__ == "__main__":
	main()