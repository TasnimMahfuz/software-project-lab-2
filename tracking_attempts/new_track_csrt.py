import cv2

# Create a video capture object to get frames from a video file or a webcam.
# If using a webcam, you can use cv2.VideoCapture(0).
video_capture = cv2.VideoCapture(0)

# Check if the video capture is working
if not video_capture.isOpened():
    print("Error: Could not open video")
    exit()

# Get the first frame
ret, frame = video_capture.read()
if not ret:
    print("Error: Could not read frame")
    exit()

# Initialize the bounding box for the object to track
# Using cv2.selectROI to let the user choose the object to track
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")

# Create the MIL tracker
tracker = cv2.TrackerMIL_create()

# Initialize the tracker with the first frame and the bounding box
tracker.init(frame, bbox)

# Process frames in a loop
while True:
    ret, frame = video_capture.read()  # Read a new frame from the video capture

    if not ret:  # If no more frames, break the loop
        break

    # Update the tracker with the current frame
    success, bbox = tracker.update(frame)

    if success:
        # Draw the bounding box if tracking is successful
        x, y, w, h = bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        # If tracking fails, display an error message
        cv2.putText(frame, "Tracking failure", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame with the bounding box
    cv2.imshow("Tracking", frame)

    # If the 'q' key is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
