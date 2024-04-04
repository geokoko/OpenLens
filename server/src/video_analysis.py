import cv2
import torch
from torchvision import transforms
import numpy as np

class VideoAnalysis():
    def __init__(self, model):
        self.model = model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_frame(self, frame):
        ''' This function processes a single frame and returns the model's prediction '''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_processed = self.transform(frame)
        frame_processed = frame_processed.unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(frame_processed)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            probabilities = probabilities.numpy()
        
        emotion_idx = np.argmax(probabilities)

        return emotion_idx, probabilities
        
    def analyze_video(self, video):
        if not video:
            raise ValueError("Video path is NULL")
    
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise IOError("Failed to open video file: " + video)
        
        frame_count = 0
        
        while 1:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}...")
            
            # Process the frame and get the detected emotion
            emotion = self.process_frame(frame)
            print(f"Detected emotion: {emotion}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        print("Video analysis completed.")
