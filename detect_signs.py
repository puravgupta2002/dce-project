import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

# Define model (same as training)
class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# Load label map
with open("model/labels.txt", "r") as f:
    label_map = {int(line.split(",")[0]): line.strip().split(",")[1] for line in f}
num_classes = len(label_map)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureNet(42, num_classes).to(device)  # 21 landmarks * 2 (x, y)
model.load_state_dict(torch.load("model/sign_language_model.pth", map_location=device))
model.eval()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

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
            if len(data) == 42:
                input_tensor = torch.tensor([data], dtype=torch.float32).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                    conf = conf.item()
                    pred = pred.item()
                    label = label_map[pred] if conf > 0.75 else "Not Recognized"
                    cv2.putText(frame, f"{label} ({conf:.2f})", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    cv2.imshow("Real-Time Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
