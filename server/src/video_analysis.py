import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from src.models.deep_emotion import Deep_Emotion

class VideoAnalysis:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize Face Detector (MTCNN)
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        
        # 2. Initialize Emotion Model
        self.label_mapping = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}
        self.model = Deep_Emotion(num_classes=len(self.label_mapping))
        
        if model_path and torch.os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded emotion model from {model_path}")
        else:
            print("Warning: No model weights found, using uninitialized model.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # 3. Define Image Preprocessing
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def detect_emotion(self, binary_data):
        """
        Processes binary MJPEG data and returns the detected emotion.
        """
        # Decode MJPEG to OpenCV image
        nparr = np.frombuffer(binary_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None

        # Convert to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Detect face
        boxes, _ = self.mtcnn.detect(pil_img)
        
        if boxes is not None:
            # Take the first detected face
            box = boxes[0]
            x, y, w, h = map(int, box)
            
            # Crop and Preprocess
            face_roi = rgb_frame[y:h, x:w]
            if face_roi.size == 0:
                return "neutral" # Fallback
                
            face_pil = Image.fromarray(face_roi)
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(face_tensor)
                _, predicted = torch.max(outputs, 1)
                emotion = self.label_mapping[predicted.item()]
                return emotion
        
        return "neutral" # Default if no face detected
