import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode = False,
                      model_complexity = 1,
                      min_detection_confidence = 0.5,
                      min_tracking_confidence= 0.75,
                      max_num_hands = 50)

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    img = cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB)

    
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks_px = [(int(l.x*img.shape[1]), int(l.y*img.shape[0])) for l in hand_landmarks.landmark]

            x_min = min(landmarks_px,key = lambda x: x[0])[0]
            y_min = min(landmarks_px, key=lambda x: x[1])[1]
            x_max = max(landmarks_px, key=lambda x :x[0])[0]
            y_max = max(landmarks_px, key = lambda x: x[1])[1]

            cv2.rectangle(img, (x_min,y_min), (x_max,y_max), (255,0,0),2)
            cv2.putText(img,'Hand!',(x_min,y_min-10),cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Image',img)

    if cv2.waitKey(1) & 0xff ==ord('q'):
        break