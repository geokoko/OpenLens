import sys
sys.path.append('../src')

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from emotion_cnn import Deep_Emotion

label_mapping = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

model = Deep_Emotion()
# Load your trained emotion detection model
model_path = '../src/model/deep_emotion_best.pth'
model_weights = torch.load(model_path)
model.load_state_dict(model_weights)
model.eval()

# Haar Cascade to detect faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define transformations for the face image
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Resize to 48x48
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        face_roi = gray[y:y+h, x:x+w]
        pil_image = Image.fromarray(face_roi)
        input_tensor = transform(pil_image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Predict emotion
        with torch.no_grad():
            output = model(input_batch)
        _, predicted = torch.max(output, 1)
        predicted_emotion = label_mapping[predicted.item()]  # Map this to actual emotion

        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
