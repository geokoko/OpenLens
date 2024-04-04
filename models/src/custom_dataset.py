from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, dataset_path, label_mapping, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.label_mapping = label_mapping
        self.images, self.labels = self.load_data()

        def load_images_and_labels(self):
            images, labels = [], []
            for label_name in os.listdir(self.dataset_path):
                class_path = os.path.join(self.dataset_path, label_name)
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    images.append(image_path)
                    labels.append(self.label_mapping[label_name])
            return images, labels
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image_path = self.images[idx]
            label = self.labels[idx]

            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return image, label

