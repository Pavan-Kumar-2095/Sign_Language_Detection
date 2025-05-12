import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


import pickle


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

with open('model2_alpha_f.pkl', 'rb') as file:
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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks and flatten them
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict the gesture
            prediction = model.predict(landmarks)
            gesture_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            # gesture_array = [1,2,3,4,5,6,7,8,9,"I love you","No","super","Yes"]
            gesture_array = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]

            # Display the gesture class and confidence
            # if  confidence > 0.80:
            cv2.putText(frame, f'Gesture: { gesture_array[gesture_class]}, Confidence: {confidence:.2f}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # cv2.putText(frame, f'Gesture: {prediction}',
            #                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            
            # else:
            #     cv2.putText(frame, f'Gesture: Unknown, Confidence:<80%',
            #                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Draw hand landmarks on the frame
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
