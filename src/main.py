import cv2
import time
from HandTracking.HandTrackingModule import HandDetector
import numpy as np

# import tensorflow as tf
from tensorflow.keras import models, layers

# from HandRecognition.resources import *
# import HandRecognition.preprocessing as preprocessing

# Change MODELS_PATH to the models' folder
MODELS_PATH = "C:/Users/basti/OneDrive - Illinois Institute of Technology/CS512-CV-Final Project/models/"
labels = ["call", "dislike", "fist", "four", "like", "mute", "no_gesture", "ok", "one", "palm", "peace",
		  "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"]


###################################
fact=1.5
wCam, hCam = int(640 * fact), int(480 * fact)

###################################
pTime = 0  # For FPS
# Set the camera and detector object
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
max_hands = 2
detector = HandDetector(maxHands=max_hands, detectionConf=0.5)
font = cv2.FONT_HERSHEY_PLAIN

# Load model (conv1d best overall)
model = models.load_model(MODELS_PATH + "model_conv1d.h5")

# Low since we don't want any overlap (probably the same hand detected multiple times)
IOU_THRESHOLD = 0.3

def compute_iou(bbox_a: np.array, bbox_b: np.array) -> float:
	# bbox: [x_min, y_min, w, h], scaled
	x_a_min = bbox_a[0]
	y_a_min = bbox_a[1]
	x_a_max = x_a_min + bbox_a[2]
	y_a_max = y_a_min + bbox_a[3]

	x_b_min = bbox_b[0]
	y_b_min = bbox_b[1]
	x_b_max = x_b_min + bbox_b[2]
	y_b_max = y_b_min + bbox_b[3]
        
	# No overlap at all => return 0
	if x_a_max < max(x_a_max, x_b_min) or x_b_max < max(x_b_max, x_a_min): return 0
        
	# Else, compute IoU
	area_a = (x_a_max - x_a_min) * (y_a_max - y_a_min)
	area_b = (x_b_max - x_b_min) * (y_b_max - y_b_min)

	x_inter_1 = max(x_a_min, x_b_min)
	y_inter_1 = max(y_a_min, y_b_min)
        
	x_inter_2 = min(x_a_max, x_b_max)
	y_inter_2 = min(y_a_max, y_b_max)
        
	area_inter = (x_inter_2 - x_inter_1) * (y_inter_2 - y_inter_1)
        
	return area_inter / (area_a + area_b - area_inter)
	

def remove_overlapping_boxes(bboxes: np.array, confidence_score: np.array):
	assert bboxes.shape[1] == 4
	assert len(bboxes) == len(confidence_score)

	boxes_to_keep = []
	for i in range(len(bboxes) - 1):
		bbox_a = bboxes[i]
		for j in range(i, len(bboxes)):
			bbox_b = bboxes[j]
			if compute_iou(bbox_a, bbox_b) > IOU_THRESHOLD:
				# get confidence score for each bbox
				boxes_to_keep.append(j if confidence_score[i] < confidence_score[j] else i)
			else:
				boxes_to_keep.append(i)
				boxes_to_keep.append(j)
	return boxes_to_keep



###################################################################################
################################  MAIN  ###########################################
###################################################################################
# def main():
frame_count = 0
global class_pred
class_pred = []
while True:
	# Find nb of hands and draw them
	success, img = cap.read()
	img, nb_Hands = detector.find_hands(img)
	# Get the image size dimensions
	h, w, c = img.shape
	# If there are hands detected:
	if nb_Hands > 0:
		bbx_list = np.array([])
		lndmrk_list = np.array([])
		conf_list = np.array([])
		# For each hand detected, get the landmarks and bbox
		for i in range(nb_Hands):
			lmList, bbox, conf = detector.find_position(img, handNo=i, draw=True)
			bbx_list = np.append(bbx_list, np.array(bbox))
			lndmrk_list = np.append(lndmrk_list, np.array(lmList))
			conf_list = np.append(conf_list, conf)
		lndmrk_list = lndmrk_list.reshape(nb_Hands, 21, 3).astype(np.int32)
		bbx_list = bbx_list.reshape(nb_Hands, 4).astype(np.int32)

		# Remove overlapping boxes
		idx_to_keep = remove_overlapping_boxes(bbx_list, conf_list) if len(bbx_list) > 1 else [0]
		# Get the bbox indexes of boxes_to_keep in bbx_list and keep only the corresponding landmarks and confidences
		bbx_list = bbx_list[idx_to_keep]
		conf_list = conf_list[idx_to_keep]
		lndmrk_list = lndmrk_list[idx_to_keep]

		# If lndmrk_list is not empty:
		if lndmrk_list.size > 0:
			lmList = lndmrk_list[:,:,1:] / np.array([w, h])
			# Predict every 10 frames to reduce latency
			if frame_count % 10 == 0:
				# Normalize the landmarks and make the prediction for the current hand
				predictions = model.predict(lmList.reshape(-1, 21, 2), verbose=False)
				class_pred = [labels[np.argmax(predictions[i])] for i in range(len(predictions))]
			for i in range(len(lndmrk_list)):
				# Print the predicted class at the top of the box for each hand
				# Keep the previous label until the next prediction
				label = "no_gesture"
				if len(class_pred) > i:
					label = class_pred[i]
				cv2.putText(img, label, (bbx_list[i][0], bbx_list[i][1]-25), font, 2, (0, 0, 255), 2)

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
