import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import keyboard



model= "model2_alpha_f.pkl"
gesture_array = ['a', 'b', 'c', 'd', "Delete", 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', "Space", 't', 'u', 'v', 'w', 'x', 'y', 'z']


    
# model= "model1_f.pkl"
# gesture_array = ["1","2","3","4","5","6","7","8","9","I love you","No","super","Yes"]


# Load the trained model
with open(model, 'rb') as file:
    model = pickle.load(file)


# #gesture array for alphabets
# while True:
#     key = keyboard.read_key()
#     if key=="a" or key=="A" :
#         model= "model2_alpha_f.pkl"
#         gesture_array = ['a', 'b', 'c', 'd', "Delete", 'e', 'f', 'g', 'h', 'i', 'j', 'k',
#                  'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', "Space", 't', 'u', 'v', 'w', 'x', 'y', 'z']
#     elif key=="1" or key=="!" :
#         model= "model1_f.pkl"
#         gesture_array = [1,2,3,4,5,6,7,8,9,"I love you","No","super","Yes"]
#     elif key=="x":
#         break



# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# For storing results
srrr = []

# Prediction delay logic
last_prediction_time = 0
prediction_delay = 1  # in seconds

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Default values
    gesture_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Optional: draw landmarks for feedback
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Prepare input
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict gesture
            prediction = model.predict(landmarks)
            gesture_class = np.argmax(prediction)
            confidence = np.max(prediction)

            # Cooldown logic to avoid repeated predictions
            current_time = time.time()
            if confidence > 0.9 and (current_time - last_prediction_time > prediction_delay):
                gesture = gesture_array[gesture_class]

                if gesture == "Delete":
                    if srrr:
                        srrr.pop()
                elif gesture == "Space":
                    if srrr:
                        srrr.append(" ")
                elif gesture.isalpha():
                    srrr.append(gesture)

                gesture_text = gesture
                last_prediction_time = current_time

    # Fixed positions and font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)          # Black background

    # Text strings
    gesture_text_display = f'Gesture: {gesture_text}'
    sentence_text_display = f'Sentence: {"".join(srrr)}'

    # Get text sizes (for background box dimensions)
    (text_w1, text_h1), _ = cv2.getTextSize(gesture_text_display, font, font_scale, thickness)
    (text_w2, text_h2), _ = cv2.getTextSize(sentence_text_display, font, font_scale, thickness)

    # Fixed Y positions (e.g., display above a 480px bottom line)
    gesture_y = 430
    sentence_y = 470

    # Background rectangles with fixed positions and padding
    cv2.rectangle(frame, (5, gesture_y - text_h1 - 5), (5 + text_w1 + 10, gesture_y + 5), bg_color, -1)
    cv2.rectangle(frame, (5, sentence_y - text_h2 - 5), (5 + text_w2 + 10, sentence_y + 5), bg_color, -1)

    # Text on top of background
    cv2.putText(frame, gesture_text_display, (10, gesture_y), font, font_scale, text_color, thickness)
    cv2.putText(frame, sentence_text_display, (10, sentence_y), font, font_scale, text_color, thickness)

    cv2.imshow('FORTHIA - Hand Gesture Recognition', frame)
    # Exit on ESC
    if cv2.waitKey(5) & 0xFF == 27:
        break
keyboard.wait()

cap.release()
cv2.destroyAllWindows()
