import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


import pickle


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


with open('test.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)
    landmarks = []
    if results.multi_hand_landmarks:
        # Flatten landmarks for all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
    #         # If fewer than two hands are detected, append zeros
    # while len(landmarks) < 126:
    #     landmarks.extend([0.0, 0.0, 0.0])

    landmarks = np.array(landmarks).reshape(1, -1)
    # gesture_array = ["A","B","D","E","F","G","H","J","K","M","N","P","Q","R","S","T","W","X","Y","Z"]
    # Predict the gesture
    if landmarks.shape[1]==126:
        prediction = model.predict(landmarks)
        gesture_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)


        cv2.putText(frame, f'Gesture: {gesture_class}, Confidence: {confidence:.2f}',(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





