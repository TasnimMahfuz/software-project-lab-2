#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:01:10 2024

@author: nafis
"""

import cv2 
import mediapipe as mp 
from google.protobuf.json_format import MessageToDict
  
# Initializing the Model 
mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=2) 
  
# Start capturing video from webcam 
cap = cv2.VideoCapture(0) 
  
while True: 
    # Read video frame by frame 
    success, img = cap.read() 
  
    # Flip the image(frame) 
    img = cv2.flip(img, 1) 
  
    # Convert BGR image to RGB image 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
  
    # Process the RGB image 
    results = hands.process(imgRGB) 
  
    # If hands are present in image(frame) 
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert landmarks to pixel coordinates
            landmarks_px = [(int(l.x * img.shape[1]), int(l.y * img.shape[0])) for l in hand_landmarks.landmark]

            # Calculate bounding box
            x_min = min(landmarks_px, key=lambda x: x[0])[0]
            y_min = min(landmarks_px, key=lambda x: x[1])[1]
            x_max = max(landmarks_px, key=lambda x: x[0])[0]
            y_max = max(landmarks_px, key=lambda x: x[1])[1]

            # Draw bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2) 

        # Both Hands are present in image(frame) 
        if len(results.multi_handedness) == 2: 
            # Display 'Both Hands' on the image 
            cv2.putText(img, 'Both Hands', (250, 50), 
                        cv2.FONT_HERSHEY_COMPLEX, 
                        0.9, (0, 255, 0), 2) 

        # If any hand present 
        else: 
            for i in results.multi_handedness: 
                # Return whether it is Right or Left Hand 
                label = MessageToDict(i)['classification'][0]['label'] 

                if label == 'Left': 
                    # Display 'Left Hand' on 
                    # left side of window 
                    cv2.putText(img, label+' Hand', 
                                (20, 50), 
                                cv2.FONT_HERSHEY_COMPLEX,  
                                0.9, (0, 255, 0), 2) 

                if label == 'Right': 
                    # Display 'Left Hand' 
                    # on left side of window 
                    cv2.putText(img, label+' Hand', (460, 50), 
                                cv2.FONT_HERSHEY_COMPLEX, 
                                0.9, (0, 255, 0), 2) 

    # Display Video and when 'q' 
    # is entered, destroy the window 
    cv2.imshow('Image', img) 
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break
