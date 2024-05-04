import cv2

# Initialize a video capture (from a webcam or video file)
video_path = "C:/Users/Nafis/Desktop/KCFpy/test.mp4"
video_capture = cv2.VideoCapture(video_path)  # Change to 0 for webcam or use a video path

# Ensure video capture is open
if not video_capture.isOpened():
    print("Error: Could not open video capture")
    exit()

# Read the first frame
ret, frame = video_capture.read()
if not ret:
    print("Error: Could not read frame")
    exit()

# Select ROI for the initial bounding box
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")

# Create the MOSSE tracker
tracker = cv2.legacy.TrackerMOSSE_create()

# Initialize the tracker with the first frame and bounding box
tracker.init(frame, bbox)

# Start tracking loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Update the tracker
    success, bbox = tracker.update(frame)

    # Draw the bounding box
    if success:
        x, y, w, h = bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Break the loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
video_capture.release()
cv2.destroyAllWindows()
