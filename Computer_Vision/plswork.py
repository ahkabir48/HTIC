import cv2
import mediapipe as mp
from cvzone.PoseModule import PoseDetector
import time

# Initialize MediaPipe Pose class and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pd = PoseDetector()

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# Initialize array
movement_data = []

while True:
    ret, img = cap.read()
    img = pd.findPose(img)
    lmlist, bbox = pd.findPosition(img)

    time.sleep(1)
    if lmlist:
        movement_data.append(lmlist)

    cv2.imshow('frame', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# print data
print("Collected Movement Data (First 5 Frames):")
for i, frame in enumerate(movement_data[:5]):
    print(f"Frame {i+1}: {frame}")