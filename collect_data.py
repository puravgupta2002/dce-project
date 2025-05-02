import cv2
import csv
import os
import mediapipe as mp

label = input("Enter label name: ")

os.makedirs("dataset", exist_ok=True)
file_path = f"dataset/{label}.csv"

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

with open(file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y])
                writer.writerow(data)

        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
