from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as transforms
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

"""Dataset used: FER 2013
"Challenges in Representation Learning: A report on three machine learning
contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,
X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
Y. Bengio. arXiv 2013."""

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
        for label_name in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, label_name)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                        try:
                            image_path = os.path.join(class_path, image_name)
                            print(image_path)
                            with Image.open(image_path) as img:
                                img.verify()
                            images.append(image_path)
                            labels.append(self.label_mapping[label_name])
                        except (IOError, UnidentifiedImageError) as e:
                            print(f"Error loading image: {image_path}: {e}")
                    else:
                        print(f"Skipped file (not an image): {image_name}")
            else:
                print (f"Warning! Directory does not exist: {class_path}")
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_class_distribution(self):
        """Return a dictionary with counts of each class"""
        return Counter(self.labels)
    
    def balance_dataset(self):
        """Balance the dataset by oversampling the minority classes"""
        paths_np = np.array(self.images).reshape(-1, 1)
        labels_np = np.array(self.labels)

        ros = RandomOverSampler(random_state=42)
        paths_resampled, labels_resampled = ros.fit_resample(paths_np, labels_np)

        return paths_resampled.flatten().tolist(), labels_resampled.tolist()
        
