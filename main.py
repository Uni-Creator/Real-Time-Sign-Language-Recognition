import torch
import numpy as np
import torch.nn as nn
import cv2  # OpenCV for capturing video
import os
import mediapipe as mp
import time
import torch.nn.functional as F


# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function for Mediapipe Detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x[:, -1, :])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# Load the trained model
def load_model(model_path, input_size, num_classes):
    model = LSTMModel(input_size=input_size, hidden_size=256, output_size=num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set to evaluation mode
    return model

# Real-time gesture recognition
def recognize_gesture(model, actions):
    cap = cv2.VideoCapture(0)
    
    sequence_length = 30  # Same as training
    sequence = []  # Store the current sequence of frames
    
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    while cap.isOpened:
        
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)

        # Draw Landmarks (Optional for display purposes)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        
        # Append the frame to the sequence
        sequence.append(keypoints)
        
        # print(len(keypoints))
        
        # If we have enough frames, make a prediction
        if len(sequence) == sequence_length:
            input_data = np.array(sequence)
            input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
            input_tensor = torch.tensor(input_data, dtype=torch.float32)

            with torch.no_grad():  # No need to track gradients during inference
                output = model(input_tensor)
                # print(output)
                prediction = output.argmax(dim=1).item()
                gesture = actions[prediction]  # Get the predicted action

            # print(f'Predicted gesture: {gesture}')
            cv2.putText(image, f'Predicted gesture: {gesture}', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)

            # Clear the sequence after prediction
            sequence.pop(0)  # Remove the first frame to keep the size consistent

        # Display the frame
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

# Usage
model_path = 'best_lstm_model.h5'  # Path to the saved model
actions = ['nothing', 'hello', 'thanks', 'iloveyou']  # Your actions
input_size = 1662  # Same as training

model = load_model(model_path, input_size, len(actions))
print(model)

recognize_gesture(model, actions)
