import sys
sys.path.append('../src')

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from emotion_cnn import Deep_Emotion
from mobile_net import MobileNet
import torchvision.transforms as transforms

#label_mapping = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
label_mapping = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((48, 48)),  # Resize to 48x48
    transforms.ToTensor(), # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])
])

model = Deep_Emotion()

# Load your trained emotion detection model
model_path = '../src/model/deep_emotion_affectnet_8_classes.pth'
model_weights = torch.load(model_path)
model.load_state_dict(model_weights)
model.eval()

# Haar Cascade to detect faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

vid_path = 'samples/1.mp4'

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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        face_roi = frame[y:y+h, x:x+w]
        pil_image = Image.fromarray(face_roi)
        input_tensor = transform(pil_image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Predict emotion
        with torch.no_grad():
            output = model(input_batch)
            global probabilities
            probabilities = torch.softmax(output, dim=1)[0]
            max_prob, predicted = torch.max(probabilities, 0)
            predicted_emotion = label_mapping[predicted.item()]  # Map this to actual emotion

        # Display the emotion
        emotion_text = f"{predicted_emotion}"
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    probs_image = np.zeros((100, 400, 3), np.uint8)
    y_offset = 15
    for idx, prob in enumerate(probabilities):
        line = f"{label_mapping[idx]}: {prob*100:.2f}%"
        cv2.putText(probs_image, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (479,479,479), 1)
        y_offset += 15

    cv2.imshow('probabilities', probs_image)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
