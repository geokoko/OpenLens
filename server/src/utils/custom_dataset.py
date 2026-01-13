from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

class CustomDataset(Dataset):
    def __init__(self, dataset_path, label_mapping, transform=None, balance_dataset=False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.label_mapping = label_mapping
        self.images, self.labels = self.load_images_and_labels()

        if balance_dataset:
            self.images, self.labels = self.balance_dataset()

    def load_images_and_labels(self):
        images, labels = [], []
        if not os.path.exists(self.dataset_path):
             print(f"Warning! Dataset path does not exist: {self.dataset_path}")
             return [], []

        for label_name in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, label_name)
            if os.path.isdir(class_path):
                if label_name not in self.label_mapping:
                    continue # Skip folders not in our mapping
                
                for image_name in os.listdir(class_path):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                        try:
                            image_path = os.path.join(class_path, image_name)
                            # Quick verify without full load
                            with Image.open(image_path) as img:
                                img.verify()
                            images.append(image_path)
                            labels.append(self.label_mapping[label_name])
                        except (IOError, UnidentifiedImageError) as e:
                            print(f"Error loading image: {image_path}: {e}")
                    else:
                        pass # Silently skip non-images
            else:
                pass
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert('L') # Convert to grayscale immediately
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Return a dummy tensor or handle appropriately (for now, let's crash so we know)
            raise e
    
    def get_class_distribution(self):
        """Return a dictionary with counts of each class"""
        return Counter(self.labels)
    
    def balance_dataset(self):
        """Balance the dataset by oversampling the minority classes"""
        if not self.images:
            return [], []
            
        paths_np = np.array(self.images).reshape(-1, 1)
        labels_np = np.array(self.labels)

        ros = RandomOverSampler(random_state=42)
        paths_resampled, labels_resampled = ros.fit_resample(paths_np, labels_np)

        return paths_resampled.flatten().tolist(), labels_resampled.tolist()
