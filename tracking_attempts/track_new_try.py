import cv2
import time

# Initialize a video capture object with a specified file path
video_path = "C:/Users/Nafis/Desktop/KCFpy/test.mp4"
video_capture = cv2.VideoCapture(video_path)

# Ensure the video capture is open
if not video_capture.isOpened():
    print("Error: Could not open video capture")
    exit()

# Get the frame rate of the video
frame_rate = video_capture.get(cv2.CAP_PROP_FPS)  # Get the frames per second (FPS)
if frame_rate == 0:
    print("Error: Could not retrieve frame rate")
    exit()

# Calculate the delay to match the video frame rate
frame_delay = int(1000 / frame_rate)  # Delay in milliseconds to achieve correct playback speed

# Read the first frame
ret, frame = video_capture.read()
if not ret:
    print("Error: Could not read frame")
    exit()

# Select the region of interest (ROI) for the initial bounding box
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")

# Create a tracker (e.g., MOSSE, KCF, etc.)
tracker = cv2.legacy.TrackerMOSSE_create()

# Initialize the tracker with the first frame and the selected bounding box
tracker.init(frame, bbox)

# Start tracking loop
while True:
    start_time = time.time()  # Record the start time for frame processing

    ret, frame = video_capture.read()  # Read a new frame
    if not ret:
        break

    # Update the tracker
    success, bbox = tracker.update(frame)

    # Draw the bounding box
    if success:
        x, y, w, h = bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), (int(y + h))), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Calculate the time taken to process the frame
    elapsed_time = time.time() - start_time

    # Wait for the remaining time to match the frame rate
    time_to_wait = max(0, frame_delay - int(elapsed_time * 1000))
    if cv2.waitKey(time_to_wait) & 0xFF == ord('q'):
        break

# Clean up
video_capture.release()
cv2.destroyAllWindows()
