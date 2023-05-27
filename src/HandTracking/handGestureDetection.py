import cv2
import time
import HandTrackingModule as htm
import numpy as np

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
detector = htm.HandDetector(maxHands=max_hands, detectionConf=0.5)
font = cv2.FONT_HERSHEY_PLAIN
# MAIN LOOP
while True:
    # 0. Find hand landmarks
    success, img = cap.read()
    img = detector.find_hands(img)
    lmList, bbox, nb_Hands = detector.find_position(img, draw=True)
    if nb_Hands > 0:
        for i in range(nb_Hands):
            lmList, bbox, nb_Hands = detector.find_position(img, handNo=i, draw=True)
            if len(lmList) != 0:
                # 1. SELECT THE MODE : VOLUME CONTROL OR MOUSE
                # 1.1 Detecting the hand shapes for each mode

                # 1.1.1 Check which fingers are up
                fingersMode = detector.fingers_up()
                # 1.1.2 If only index is up or only index and middle fingers are up : Mouse Control
                if (fingersMode[3] == fingersMode[4] == 0) and (
                        (fingersMode[0] == fingersMode[1] == 1) or (fingersMode[0] == fingersMode[1] == fingersMode[2] == 1)):
                    # if (fingersMode[0] == fingersMode[3] == fingersMode[4] == 0) and ((fingersMode[1] == 1) or (fingersMode[1] == fingersMode[2] == 1)):
                    # 2. MODE 1 : MOUSE CONTROL
                    # Print the mode name at the top of the box
                    cv2.putText(img, "Mode: Mouse Control", (bbox[0], bbox[1]-25), font, 2, (0, 255, 0), 2)
                    # print(x1,y1,x2,y2)
                    # 2.2. Check which fingers are up
                    fingers = detector.fingers_up()
                else:
                    if (fingersMode[2] == fingersMode[3] == 1):
                        cv2.putText(img, "Mode: Detection mode", (bbox[0], bbox[1]-25), font, 2, (0, 255, 0), 2)

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
