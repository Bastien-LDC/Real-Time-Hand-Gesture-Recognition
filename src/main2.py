import cv2
import time
from HandTracking.HandTrackingModule import HandDetector
import numpy as np

import tensorflow as tf
from tensorflow.keras import models, layers
import tensorflow.keras.backend as K
from tensorflow.keras.utils import img_to_array

# from HandRecognition.resources import *
# import HandRecognition.preprocessing as preprocessing

# Change MODELS_PATH to the models' folder
MODELS_PATH = "C:/Users/basti/OneDrive - Illinois Institute of Technology/CS512-CV-Final Project/models/"
labels = ["call", "dislike", "fist", "four", "like", "mute", "no_gesture", "ok", "one", "palm", "peace",
		  "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"]


###################################
fact=1.5
wCam, hCam = int(640 * fact), int(480 * fact)
IMG_WIDTH = 1920 
IMG_HEIGHT = 1080

###################################
pTime = 0  # For FPS
# Set the camera and detector object
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
max_hands = 2
detector = HandDetector(maxHands=max_hands, detectionConf=0.5)
font = cv2.FONT_HERSHEY_PLAIN


################ CUSTOM LOSS FUNCTIONS  ################
mse = tf.keras.losses.MeanSquaredError()
cross_ent = tf.keras.losses.CategoricalCrossentropy()
huber = tf.keras.losses.Huber()
acc = tf.keras.metrics.Accuracy()
cosine_sim = tf.keras.metrics.CosineSimilarity(axis=1)
num_labels = None

_max = tf.math.maximum
_min = tf.math.minimum

def compute_iou(y_true, y_pred):
	_shape = K.shape(y_pred)
	total_iou = 0.0
	for i in range(_shape[0]):
		bbox_1 = y_true[i]
		bbox_2 = y_pred[i]
		x_1 = _max(bbox_1[0], bbox_2[0])
		y_1 = _max(bbox_1[1], bbox_2[1])
		x_2 = _min(bbox_1[2], bbox_2[2])
		y_2 = _min(bbox_1[3], bbox_2[3])
		inter_area = _max(0.0, x_2 - x_1 + 1.0) * _max(0.0, y_2 - y_1 + 1.0)

		bbox_1_area = (bbox_1[2] - bbox_1[0] + 1.0) * (bbox_1[3] - bbox_1[1] + 1.0)
		bbox_2_area = (bbox_2[2] - bbox_2[0] + 1.0) * (bbox_2[3] - bbox_2[1] + 1.0)
		total_iou += inter_area / (bbox_1_area + bbox_2_area - inter_area)
	return total_iou / tf.cast(_shape[0], tf.float32)

def custom_metric(y_true, y_pred):
	# Classification
	if y_pred.shape[1] == num_labels:
		# return acc(y_true, y_pred)
		return acc(y_true, y_pred)	
	# Landmarks
	if y_pred.shape[1] == 21:
		return cosine_sim(y_true, y_pred)	
	# Bboxes
	else:
		return compute_iou(y_true, y_pred)

def custom_loss(y_true, y_pred):
	# Classification
	if y_pred.shape[1] == num_labels:
		return cross_ent(y_true, y_pred)
	# Landmarks
	if y_pred.shape[1] == 21:
		return huber(y_true, y_pred)	
	# Bboxes
	return mse(y_true, y_pred)


################  LOAD MODELS  ################
# Load model for landmark and bbox detection
detector_name = "hand_detection_mobilenet.h5"
detection_model = models.load_model(MODELS_PATH + detector_name, custom_objects={'custom_loss': custom_loss, 'custom_metric': custom_metric})
# Load model for gesture recognition
gesture_name = "model_conv1d.h5"
gesture_model = models.load_model(MODELS_PATH + gesture_name)

SQUARE_IMG = True # Set to True if using a MobileNet pre-trained model
CLASSIFER = True # Set to True if using the classifier model (bbox, gesture), False if using only the detector (bbox, landmarks) + (gesture) models
# Classifier that predicts bbox and gesture
classifier_name = "hand_detection_mobilenet_classifier_finetuned.h5"
classifier_model = models.load_model(MODELS_PATH + classifier_name, custom_objects={'custom_loss': custom_loss, 'custom_metric': custom_metric})


def process_img(image, square_img: bool=False):
	"""
	Process the image to be used for the model.
	Args:
		img: The cv2 image object to process.
		square_img: Whether to resize the image to 224x224 (MobileNet pre-trained model).
	Returns:
		The processed image.
	"""
	img = image.copy()
	img = img_to_array(img) / 255.0		
	# resize images to same size each
	width, height, _ = img.shape

	if width > IMG_WIDTH or height > IMG_HEIGHT:
		img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
		# Downsample for training (my computer can't handle training using Full HD resolution images)
		img = cv2.pyrDown(img)
		img = cv2.pyrDown(img)

	if width > IMG_WIDTH // 2 or height > IMG_HEIGHT // 2:
		img = cv2.resize(img, (IMG_HEIGHT // 2, IMG_WIDTH // 2))
		img = cv2.pyrDown(img)

	else: img = cv2.resize(img, (IMG_HEIGHT // 4, IMG_WIDTH // 4))
	# MobileNet pre-trained requires square img, low quality...
	if square_img: img = cv2.resize(img, (224, 224))

	return np.array([img])


def draw_landmarks_bbox(img, label: str="", landmarks: np.array=None, bboxes: np.array=None, draw_bbox: bool=True, draw_landmarks: bool=True):
    """Draw the landmarks and bounding boxes on the image.
    Args:
        img: The cv2 image object to draw on.
		label: The label of the image.
		landmarks: The list of landmarks to draw.
		bboxes: The list of bounding boxes to draw.
	Returns:
		The image with the landmarks and bounding boxes drawn on it.
    """
    # Reduce the image size to 1/4
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img_h, img_w, _ = img.shape
    # Draw the bounding box on the image
    if draw_bbox or bboxes is not None:
        for box in bboxes:
            x, y, w, h = int(box[0]*img_w), int(box[1]*img_h), int(box[2]*img_w), int(box[3]*img_h)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-25), font, 2, (0, 0, 255), 2)
    # Draw the landmarks and lines between the landmarks on the image
    lm_edges = [(0, 1), (0, 17), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20)]
    if draw_landmarks or landmarks is not None:
        for lm in landmarks:
            for edge in lm_edges:
                x1, y1 = int(lm[edge[0]][0]*img_w), int(lm[edge[0]][1]*img_h)
                x2, y2 = int(lm[edge[1]][0]*img_w), int(lm[edge[1]][1]*img_h)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            for coord in lm:
                x, y = int(coord[0]*img_w), int(coord[1]*img_h)
				# Adapt the size of the circle according to the box area
                lm_size = int((w*h)**0.5/50)
                cv2.circle(img, (x, y), lm_size, (255, 0, 255), cv2.FILLED)
				# cv2.putText(img, str(i), (x, y), font, 1, (255, 512, 0), 2)
    return img


##### NOTE: This script works only for 1 hand detection as the models were trained for detecting only one hand.
###################################################################################
################################  MAIN  ###########################################
###################################################################################
# def main():
frame_count = 0
global bbox_cache, label_cache
bbox_cache = []
label_cache = ""
while True:
	# Find nb of hands and draw them
	success, img = cap.read()
	img, nb_Hands = detector.find_hands(img, draw=False)
	# Get the image size dimensions
	h, w, c = img.shape
	# If there are hands detected:
	if nb_Hands > 0:
		if CLASSIFER:
			# You can adapt the number of predictions for every n frames to reduce the latency of the video
			if frame_count % 2 == 0:
				bbox, class_pred = classifier_model.predict(process_img(img, square_img=SQUARE_IMG), verbose=False)
				label = labels[np.argmax(class_pred)]
				bbox_cache, label_cache = bbox, label
			img = draw_landmarks_bbox(img, label=label_cache, bboxes=bbox_cache, draw_landmarks=False)
		else:
			bbox, lmList = detection_model.predict(process_img(img, square_img=SQUARE_IMG), verbose=False)
			# If lmList is not empty:
			if lmList.size > 0 and bbox.size > 0:
				# Predict every 10 frames to reduce latency
				if frame_count % 10 == 0:
					# Normalize the landmarks and make the prediction for the current hand
					class_pred = gesture_model.predict(lmList, verbose=False)
					label = labels[np.argmax(class_pred)]
				img = draw_landmarks_bbox(img, label=label, landmarks=lmList, bboxes=bbox, draw_landmarks=False)

	frame_count = frame_count + 1 if frame_count < 100 else 0

	# 4. Frame rate
	cTime = time.time()
	fps = 1 / (cTime - pTime)
	pTime = cTime
	cv2.putText(img, f"FPS: {int(fps)}", (10, 40), font, 2, (255, 0, 0), 3)

	# 5. Display Image
	cv2.imshow("Image", img)
	if cv2.waitKey(2) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
