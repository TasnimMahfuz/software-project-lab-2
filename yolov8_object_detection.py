# import cv2
# from ultralytics import YOLO
#
# # Load the pre-trained YOLOv8 tiny model
# model = YOLO('yolov8n.pt')  # 'n' stands for nano, which is a very small model
#
# # Class names for COCO dataset
# class_names = {0: 'person', 39: 'bottle', 67: 'cell phone'}
#
# # Initialize the webcam
# cap = cv2.VideoCapture(0)
#
# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     if not ret:
#         print("Error: Failed to capture image.")
#         break
#
#     # Perform inference on the frame
#     results = model(frame)
#
#     # Iterate over detected objects
#     for result in results:
#         boxes = result.boxes  # Access the bounding boxes
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates
#             conf = box.conf[0]  # Confidence score
#             cls = int(box.cls[0])  # Class label
#
#             # Check if the detected object is one of the specified classes
#             if cls in class_names:
#                 label = f'{class_names[cls]} {conf:.2f}'
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#     # Display the resulting frame
#     cv2.imshow('Specific Object Detection', frame)
#
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the webcam and close windows
# cap.release()
# cv2.destroyAllWindows()



import numpy as np
import cv2
from collections import deque
import mediapipe as mp
from ultralytics import YOLO
from utils.utils_v2 import get_idx_to_coordinates, rescale_frame

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Load the pre-trained YOLOv8 tiny model
model = YOLO('yolov8n.pt')  # 'n' stands for nano, which is a very small model

# Define zones based on shoulder positions
def define_zones(image, left_shoulder, right_shoulder):
    height, width, _ = image.shape
    body_width = abs(right_shoulder[0] - left_shoulder[0])
    body_height = height // 2  # Increased height for the boxes

    gap = 20  # Adjust this value to increase the gap between the boxes

    # Green zone covering the body
    front_zone = (
        left_shoulder[0],
        left_shoulder[1],
        right_shoulder[0],
        left_shoulder[1] + body_height
    )

    # Red zone to the right of the body, not touching the green zone
    right_zone = (
        right_shoulder[0] + body_width + gap,
        right_shoulder[1],
        right_shoulder[0] + 2 * body_width + gap,
        right_shoulder[1] + body_height
    )

    return front_zone, right_zone

# Function to check if point is in a zone
def is_in_zone(point, zone):
    x, y = point
    x1, y1, x2, y2 = zone
    return x1 < x < x2 and y1 < y < y2

def main():
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
    hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
    cap = cv2.VideoCapture(0)
    pts = deque(maxlen=64)
    count = 0
    hand_in_red_zone = False

    while cap.isOpened():
        idx_to_coordinates = {}
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hand = hands.process(image)
        results_pose = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0]))
            right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]),
                              int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0]))

            # Define zones
            front_zone, right_zone = define_zones(image, left_shoulder, right_shoulder)

            # Draw shoulder landmarks
            cv2.circle(image, left_shoulder, 5, (0, 255, 0), -1)
            cv2.circle(image, right_shoulder, 5, (0, 255, 0), -1)

            hands_positions = []
            if results_hand.multi_hand_landmarks:
                for hand_landmarks in results_hand.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=hand_landmark_drawing_spec,
                        connection_drawing_spec=hand_connection_drawing_spec)
                    idx_to_coordinates = get_idx_to_coordinates(image, results_hand)
                    if 8 in idx_to_coordinates:  # Index Finger
                        hands_positions.append(idx_to_coordinates[8])

            # Perform object detection
            results = model(image)
            detected_objects = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    if cls in [39, 67]:  # Only cell phone (67) and bottle (39)
                        detected_objects.append((cls, (x1, y1, x2, y2)))
                        label = f'{model.names[cls]} {conf:.2f}'
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            hand_and_object_in_red_zone = False
            if len(hands_positions) > 0:
                for hand_position in hands_positions:
                    hand_in_right_zone = is_in_zone(hand_position, right_zone)

                    # Check if any detected object is in the right zone along with the hand
                    object_in_right_zone = any(is_in_zone(((x1 + x2) // 2, (y1 + y2) // 2), right_zone) for _, (x1, y1, x2, y2) in detected_objects)

                    if hand_in_right_zone and object_in_right_zone:
                        hand_and_object_in_red_zone = True
                        break  # No need to check other hands if one is already meeting the condition

            if hand_and_object_in_red_zone and not hand_in_red_zone:
                count += 1
                hand_in_red_zone = True
                print(f"Product count: {count}")
            elif not hand_and_object_in_red_zone:
                hand_in_red_zone = False

            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                thick = int(np.sqrt(len(pts) / float(i + 1)) * 4.5)
                cv2.line(image, pts[i - 1], pts[i], (0, 255, 0), thick)

            # Draw zones
            cv2.rectangle(image, front_zone[:2], front_zone[2:], (0, 255, 0), 2)
            cv2.rectangle(image, right_zone[:2], right_zone[2:], (0, 0, 255), 2)
            cv2.putText(image, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Res", rescale_frame(image, percent=130))
        if cv2.waitKey(5) & 0xFF == 27:
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
