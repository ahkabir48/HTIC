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

frame_count = 0
restless_count = 0
prev_x = 0
prev_y = 0

while True:
    ret, img = cap.read()
    img = pd.findPose(img)
    lmlist, bbox = pd.findPosition(img)

    time.sleep(1)
    if lmlist:
        movement_data.append(lmlist)
        # print(f" Frame: {movement_data}\n\n")

    frame = movement_data[frame_count]
    frame_count += 1
    # analyze points on face and shoulder for movement
    for point in frame[0:12]:
        if frame_count == 1:
            prev_x = point[0]
            prev_y = point[1]
        else:
            x = point[0]
            y = point[1]
            if abs(prev_x - x) > 20 or abs(prev_y - y) > 20:
                restless_count += 1
                prev_x = x
                prev_y = y
                break
            prev_x = x
            prev_y = y
    
    # track instances of movement, if more than twice, send alert
    if restless_count > 2:
        print(f"Patient motion detected!!")
        restless_count = 0

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
