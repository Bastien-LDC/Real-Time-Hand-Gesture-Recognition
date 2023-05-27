import os
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from keras.utils import img_to_array, load_img

def load_data(dir_annotations: str, bbox=True):
	annotations = []

	# Load annotation files
	for filename in os.listdir(dir_annotations):
		json_data = json.load(open(f"{dir_annotations}/{filename}", 'r'))
		for e in json_data.items():
			# e: (key, value)

			# Skip samples with no landmarks or box coordinates
			if len(e[1]['landmarks'][0]) == 0 or len(e[1]['bboxes'][0]) != 4: continue

			annotations.append(e[1])

	num_landmarks = len(annotations[0]['landmarks'][0])

	# Input x: [[box coordinates], [landmarks]]
	# Target label y
	x = []
	y = []

	for annot in annotations:
		if bbox:
			x.append(
				(
					np.asarray(annot["bboxes"][0], dtype=np.float32), 
					np.asarray(annot["landmarks"][0], dtype=np.float32)
				)
			)
		else:
			x.append(np.asarray(annot["landmarks"][0], dtype=np.float32))

		y.append(annot["labels"][0])


	y = np.array(y)
	x = np.array(x)

	labels = np.unique(y)

	return x, y, num_landmarks, labels


	
def load_data_leading_hand(dir_annotations: str, bbox=True):
	annotations = []

	for filename in os.listdir(dir_annotations):
		json_data = json.load(open(f"{dir_annotations}/{filename}", 'r'))
		for e in json_data.items():
			# e: (key, value)

			# Skip samples with no landmarks or box coordinates
			if len(e[1]['landmarks'][0]) == 0 or len(e[1]['bboxes'][0]) != 4: continue

			annotations.append(e[1])

	num_landmarks = len(annotations[0]['landmarks'][0])

	# Input x: [[box coordinates], [landmarks]]
	# Target label y
	x = []
	y = []

	for annot in annotations:
		# Two hands detected => keep 
		if len(annot["landmarks"]) != 2: continue
		if len(annot["landmarks"][0]) == 0 or len(annot["landmarks"][0]) != len(annot["landmarks"][1]): continue

		if bbox:
			x.append(
				(
					np.asarray(annot["bboxes"], dtype=np.float32), 
					np.asarray(annot["landmarks"], dtype=np.float32)
				)
			)
		else:
			x.append(np.asarray(annot["landmarks"], dtype=np.float32))

		# 1: right hand
		# 0: left hand
		y.append(int(str(annot["leading_hand"]).lower() == "right"))


	y = np.array(y)
	x = np.array(x)

	labels = np.unique(y)

	return x, y, num_landmarks, labels


def load_data_hand_detection(dir_annotations: str, dir_img: str, num_landmarks=21, with_labels=False, just_labels=False):
	# Each element contains landmarks [right_hand, left_hand], one possibly null if just one hand is detected
	x = [] # Images as array
	y = [] # [[box coordinates], [landmarks]]
	labels = []

	for filename in os.listdir(dir_annotations):
		json_data = json.load(open(f"{dir_annotations}/{filename}", 'r'))

		# Get all the images of the subset (remove .jpg)
		img_files = list(
			map(
				lambda x: x.split('.')[0], 
				os.listdir(f"{dir_img}/{filename.split('.')[0]}")
			)
		)
			 
		for e in json_data.items():
			# e: (key, value)

			# If the image is not in subset => skip
			if e[0] not in img_files: continue

			# Skip samples with no landmarks or box coordinates
			if len(e[1]['landmarks'][0]) == 0 or len(e[1]['bboxes'][0]) != 4: continue

			# One hand
			if len(e[1]['landmarks']) == 1 and len(e[1]['landmarks'][0]) > 0 and len(e[1]["bboxes"][0]) == 4:	
				x.append(f"{dir_img}/{filename.split('.')[0]}/{e[0]}.jpg")

				y.append(
					[
						np.array(e[1]["bboxes"][0]).astype(np.float32),
						np.array(e[1]['landmarks'][0]).astype(np.float32)
					]
				)
				labels.append(e[1]["labels"][0])
			

	labels = np.array(labels)
	unique_labels = np.unique(labels.flatten())

	if with_labels or just_labels:
		# One hot encoding of the labels
		encoder = OneHotEncoder(sparse=False)
		encoded_labels = labels.copy()

		encoded_labels = encoder.fit_transform(encoded_labels.reshape(-1, 1))
		encoded_labels = encoded_labels.reshape(-1, len(unique_labels))

		if just_labels: return encoded_labels, unique_labels

		y = [(y[i][0], encoded_labels[i]) for i in range(len(encoded_labels))]

	y = np.array(y, dtype=object)
	x = np.array(x)

	if with_labels: return x, y, unique_labels

	return x, y