import numpy as np
import cv2
import sys
from time import time

import kcftracker  # Make sure this is imported correctly

# Global state variables
selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0
interval = 1
duration = 0.01

# Mouse callback function
def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h
    
    if event == cv2.EVENT_LBUTTONDOWN:
        selectingObject = True
        onTracking = False
        ix, iy = x, y
        cx, cy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if selectingObject:
            cx, cy = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        selectingObject = False
        if abs(x - ix) > 10 and abs(y - iy) > 10:
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            initTracking = True
        else:
            onTracking = False
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if w > 0:
            ix, iy = x - w / 2, y - h / 2
            initTracking = True

# Main script
if __name__ == '__main__':
    if len(sys.argv) == 1:
        cap = cv2.VideoCapture(0)  # Default camera
    elif len(sys.argv) == 2:
        if sys.argv[1].isdigit():
            cap = cv2.VideoCapture(int(sys.argv[1]))  # Camera index
        else:
            cap = cv2.VideoCapture(sys.argv[1])  # Video file
            interval = 30
    else:
        raise ValueError("Too many arguments")

    tracker = kcftracker.KCFTracker(True, True, True)  # HOG, fixed_window, multiscale

    cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', draw_boundingbox)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if selectingObject:
            cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
        elif initTracking:
            cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)

            tracker.init([ix, iy, w, h], frame)  # Initialize tracker

            initTracking = False
            onTracking = True
        elif onTracking:
            t0 = time()
            boundingbox = tracker.update(frame)  # Get the bounding box
            boundingbox = list(map(int, boundingbox))  # Convert to a list
            
            # Draw the bounding box
            cv2.rectangle(frame, 
                          (boundingbox[0], boundingbox[1]), 
                          (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), 
                          (0, 255, 255), 
                          1)
            
            duration = 0.8 * duration + 0.2 * (time() - t0)  # Calculate duration for FPS estimation
            cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4], 
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 255), 2)  # Display FPS

        cv2.imshow('tracking', frame)  # Show the tracking output

        c = cv2.waitKey(interval) & 0xFF  # Get key press
        if c == 27 or c == ord('q'):  # Exit on ESC or 'q'
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close OpenCV windows
