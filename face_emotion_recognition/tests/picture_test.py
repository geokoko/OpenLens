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
model_path = '../src/model/deep_emotion_affectnet_8_classess.pth'
model_weights = torch.load(model_path)
model.load_state_dict(model_weights)
model.eval()

# Haar Cascade to detect faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img_path = 'samples/h1.jpg'

# Convert to grayscale for face detection
frame = cv2.imread(img_path)
if frame is None:
    print("Cannot open image")
    exit()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        probabilities = probabilities.numpy()
    
    emotion_idx = np.argmax(probabilities)
    emotion = label_mapping[emotion_idx]
    print(f"Detected emotion: {emotion}")

    cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()