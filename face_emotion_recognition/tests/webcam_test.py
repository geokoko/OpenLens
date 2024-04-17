import sys
sys.path.append('../src')

from facenet_pytorch import MTCNN
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from emotion_cnn_for_7_classes import Deep_Emotion
from mobile_net import MobileNet
import torchvision.transforms as transforms

label_mapping = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
#label_mapping = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((48, 48)),  # Resize to 48x48
    transforms.ToTensor(), # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])
])

model = Deep_Emotion()

# Load your trained emotion detection model
model_path = '../src/model/deep_emotion-1.pt'
model_weights = torch.load(model_path)
model.load_state_dict(model_weights)
model.eval()

# detect faces
mtcnn = MTCNN(keep_all=True, device='cpu')

vid_path = 'samples/2.mp4'

# Initialize webcam
cap = cv2.VideoCapture(vid_path)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

probabilities = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)
    
    if boxes is not None:
        for box in boxes:
            x, y, w, h = map(int, box)

            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)

            # Crop the face
            face_roi = rgb_frame[y:h, x:w]
            face = Image.fromarray(face_roi).convert('L')
            face = transform(face).unsqueeze(0)

            # Predict emotion
            with torch.no_grad():
                output = model(face)
                probabilities = torch.softmax(output, dim=1)[0]
                max_prob, predicted = torch.max(probabilities, 0)
                predicted_emotion = label_mapping[predicted.item()]  # Map this to actual emotion

            # Display the emotion
            emotion_text = f"{predicted_emotion}"
            cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    if probabilities is not None:
        probs_image = np.zeros((400, 400, 3), np.uint8)
        y_offset = 15
        for idx, prob in enumerate(probabilities):
            line = f"{label_mapping[idx]}: {prob*100:.2f}%"
            cv2.putText(probs_image, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (479,479,479), 1)
            y_offset += 30

    cv2.imshow('probabilities', probs_image)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
