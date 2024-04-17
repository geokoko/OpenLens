import sys
sys.path.append('../src')

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
#from emotion_cnn_for_7_classes import Deep_Emotion
from emotion_cnn import Deep_Emotion
from mobile_net import MobileNet
import os
import torchvision.transforms as transforms

#label_mapping = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
label_mapping = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((48, 48)),  # Resize to 48x48
    transforms.ToTensor(), # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#model = Deep_Emotion()
model = MobileNet()

# Load your trained emotion detection model
model_path = '../src/model/mobile_net.pth'
model_weights = torch.load(model_path)
model.load_state_dict(model_weights)
model.eval()

# Haar Cascade to detect faces

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

incorrect = 0
correct = 0

for emot in label_mapping.values():
    correct = 0
    incorrect = 0
    if emot == 'contempt':
        continue
    for img_path in os.listdir(f'samples/{emot}'):

        # Convert to grayscale for face detection
        frame = cv2.imread(os.path.join(f'samples/{emot}', img_path))
        if frame is None:
            print("Cannot open image")
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Create a mini-batch as expected by the model

        input_batch = transform(Image.fromarray(gray_frame)).unsqueeze(0)
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            probabilities = probabilities.numpy()
                
        emotion_idx = np.argmax(probabilities)
        emotion = label_mapping[emotion_idx]
        if emotion != emot:
            incorrect += 1
        if emotion == emot:
            correct += 1

            #cv2.imshow('frame', frame)

    print(f"Incorrect {emot} emotions: {incorrect}")
    print(f"Correct {emot} emotions: {correct}")
cv2.waitKey(0)
cv2.destroyAllWindows()
