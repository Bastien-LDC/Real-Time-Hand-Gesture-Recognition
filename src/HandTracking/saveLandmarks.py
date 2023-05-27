import cv2
import time
import HandTrackingModule as htm
import numpy as np
import os
import glob
import json
# Import tqdm for notebook progress bar
from tqdm import tqdm

###################################
fact=1.5
wCam, hCam = int(640*fact), int(480*fact)
###################################
pTime = 0  # For FPS
# Set the camera and detector object
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
max_hands = 2
detector = htm.HandDetector(maxHands=max_hands, detectionConf=0.6)
font = cv2.FONT_HERSHEY_PLAIN

IMG_PATH = "C:/Users/basti/OneDrive - Illinois Institute of Technology/CS512-CV-Final Project/data/images"
IMG_SAVE_PATH = "C:/Users/basti/OneDrive - Illinois Institute of Technology/CS512-CV-Final Project/data/images_mediapipe"
ANNOT_SAVE_PATH = "C:/Users/basti/OneDrive - Illinois Institute of Technology/CS512-CV-Final Project/data/annot_mediapipe"
PREFIX = "train_val_"
labels = ["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"]

# os.makedirs(os.path.dirname(IMG_SAVE_PATH+"/"), exist_ok=True)
os.makedirs(os.path.dirname(ANNOT_SAVE_PATH + "/"), exist_ok=True)

SAMPLE_SIZE = 2500

# For all images in the label folders, pass them through the detector and save the image with the landmarks 
for label in tqdm(labels, desc="Labels"):
	curr_label_sample = 0
	# Create the folder if it doesn't exist
	# os.makedirs(os.path.dirname(f"{IMG_SAVE_PATH}/{label}/"), exist_ok=True)
	img_list = glob.glob(f"{IMG_PATH}/{PREFIX+label}/*.jpg")
	annot = {}
	for i in tqdm(range(0, len(img_list)), desc="Images", unit="imgs"):
		if curr_label_sample >= SAMPLE_SIZE:
			break
		img = cv2.imread(img_list[i])    
		img_id = img_list[i].split("\\")[-1].split(".")[0]
		# Get the image size dimensions
		h, w, c = img.shape
		img, nb_Hands = detector.find_hands(img)
		if nb_Hands > 0:
			bbx_list = []
			lndmrk_list = []
			conf_list = []
			for j in range(nb_Hands):
				lmList, bbox, conf = detector.find_position(img, handNo=j, draw=True)
				if len(lmList) != 0:
					# Compute wb and hb
					wb = bbox[2] - bbox[0] + 1
					hb = bbox[3] - bbox[1] + 1

					# Replace the last 2 bbox coordinates with wb and hb
					bbox[2], bbox[3] = wb, hb

					# Normalize the landmarks and bbox coordinates to the image size
					lmList = np.array(lmList)[:,1:] / np.array([w, h])
					bbox = np.array(bbox) / np.array([w, h, w, h])

					# Append the landmarks and bbox to the list
					lndmrk_list.append(lmList.tolist())
					bbx_list.append(bbox.tolist())
					conf_list.append(conf)
					# cv2.imwrite(f"{IMG_SAVE_PATH}/{label}/{curr_label_sample}.jpg", img)

			annot[img_id] = {"nb_hands": nb_Hands, "labels": label, "bboxes": bbx_list, "landmarks": lndmrk_list, "conf": conf_list}
			curr_label_sample += 1
		# else:
		#     print(f"Label {label} : {i} : No hands detected")
			# cv2.imwrite(f"{IMG_SAVE_PATH}/{label}/{i}.jpg", img)
	# Create the annotation file for the current label and save it
	with open(f"{ANNOT_SAVE_PATH}/{label}.json", "w") as f:
		json.dump(annot, f)

	print(f"Label {label} : {curr_label_sample}/{len(img_list)} images saved")
	print(f"{ANNOT_SAVE_PATH}/{label}.json saved")