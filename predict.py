import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os


print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)
# loop over the images that we'll be testing using our bounding box
# regression model
	# load the input image (in Keras format) from disk and preprocess
	# it, scaling the pixel intensities to the range [0, 1]

cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()

frame = cv2.imread("./test.jpg")
image = cv2.resize(frame, (224, 224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)
preds = model.predict(image)[0]
(startX, startY, endX, endY) = preds

# image = cv2.imread("./test.jpg")
image = cv2.resize(frame, (224, 224))
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]
startX = int(startX * w)
startY = int(startY * h)
endX = int(endX * w)
endY = int(endY * h)
cv2.rectangle(image, (startX, startY), (endX, endY),
    (0, 255, 0), 2)
cv2.imshow("Output", image)
k = cv2.waitKey(0)
# if k == ord('n'):
#     continue
# if k == ord('q'):
#     break
