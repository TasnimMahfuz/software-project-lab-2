import numpy as np
import cv2
import mediapipe as mp
import time
from utils.utils_v2 import get_idx_to_coordinates, rescale_frame

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

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
    count = 0
    hands_in_red_zone = False

    working_time = 0
    idle_time = 0
    last_update_time = time.time()
    last_movement_time = time.time()
    prev_hand_positions = []

    while cap.isOpened():
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

            # Initialize any_hand_in_red_zone
            any_hand_in_red_zone = False

            # Check if any hand is in the red zone
            if len(hands_positions) >= 1:
                any_hand_in_red_zone = any(is_in_zone(hand_position, right_zone) for hand_position in hands_positions)
                if any_hand_in_red_zone and not hands_in_red_zone:
                    count += 1
                    hands_in_red_zone = True
                    print(f"Product count: {count}")
                elif not any_hand_in_red_zone:
                    hands_in_red_zone = False

            current_time = time.time()
            elapsed_time = current_time - last_update_time

            # Determine if hands are moving or idle
            hand_movement = False
            if prev_hand_positions:
                hand_movement = any(np.linalg.norm(np.array(prev) - np.array(curr)) > 20 for prev, curr in zip(prev_hand_positions, hands_positions))

            if hand_movement or any_hand_in_red_zone:
                working_time += elapsed_time
                last_movement_time = current_time  # Update the last movement time
            else:
                # Check if it's been 2 seconds since the last movement
                if current_time - last_movement_time > 2:
                    idle_time += elapsed_time

            prev_hand_positions = hands_positions if hands_positions else prev_hand_positions
            last_update_time = current_time

            # Draw zones
            cv2.rectangle(image, front_zone[:2], front_zone[2:], (0, 255, 0), 2)
            cv2.rectangle(image, right_zone[:2], right_zone[2:], (0, 0, 255), 2)
            cv2.putText(image, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Working Time: {working_time:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Idle Time: {idle_time:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Res", rescale_frame(image, percent=130))
        if cv2.waitKey(5) & 0xFF == 27:
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
