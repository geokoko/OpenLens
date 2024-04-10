import torch.optim as optim
import torch.nn as nn
from emotion_cnn import Deep_Emotion
from mobile_net import MobileNet
import torch
from generate_plot import generate_plot
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Check if CUDA is available
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda" if train_on_gpu else "cpu")
print(f'Using device: {device}')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from torchvision import transforms, datasets, utils
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import importlib
from custom_dataset import CustomDataset
from collections import Counter

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(), # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485], std=[0.229]) # Normalize the pixel values
])

# Define the label mapping
label_mapping = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

# Load the datasets
train_dataset_fer = CustomDataset('../data/fer2013/images/train/', label_mapping=label_mapping, transform=transform)
val_dataset_fer = CustomDataset('../data/fer2013/images/validation/', label_mapping=label_mapping, transform=transform)

train_dataset_affectnet = CustomDataset('../data/affectnet/train/', label_mapping=label_mapping, transform=transform)
val_dataset_affectnet = CustomDataset('../data/affectnet/validation/', label_mapping=label_mapping, transform=transform)

train_dataset_final = train_dataset_affectnet
val_dataset_final = val_dataset_affectnet
print(f"Training samples: {len(train_dataset_final)}")
print(f"Validation samples: {len(val_dataset_final)}")

# Get combined train_dataset distribution

"""for dataset in  [train_dataset_fer, train_dataset_affectnet]:
    dataset_distribution = dataset.get_class_distribution()
    combined_distribution.update(dataset_distribution)

combined_distribution = dict(combined_distribution)
print(f"Combined training dataset distribution: {combined_distribution}")
"""


# Calculate class weights for loss function
total_samples = len(train_dataset_final)
class_weights = {class_id : total_samples/ ( 7 * dict(train_dataset_final.get_class_distribution())[class_id] ) for class_id in dict(train_dataset_final.get_class_distribution()).keys()}

weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float)
# Create the data loaders
train_loader = DataLoader(train_dataset_final, batch_size=32, shuffle=True, num_workers=4)
print(f'Number of batches {len(train_loader)}')
val_loader = DataLoader(val_dataset_final, batch_size=32, shuffle=False, num_workers=4)
print(f'Number of batches {len(val_loader)}')

# Initialize the model, optimizer and loss function
# model = Deep_Emotion().to(device)
model = Deep_Emotion().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device)) # Cross-entropy loss for classification problems
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
best_accuracy = 0.0
best_val_loss = np.inf
patience = 10
epochs_without_improvement = 0

train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []
# Training loop
epochs = 300
for epoch in range(epochs):
    # Set predictions and labels lists
    print(f"Starting {epoch+1} epoch")
    all_preds = [] 
    all_labels = []

    model.train()
    train_loss = 0.0
    validation_loss = 0.0
    train_correct = 0.0
    validation_correct = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) # transfer image data and labels to GPU
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs,1)
        train_correct += torch.sum(preds == labels.data)
    
    train_loss /= len(train_loader)
    print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}')
    

    # Validation loop
    model.eval()
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device) # tranfer image data and labels to GPU

        val_outputs = model(images)
        val_loss = criterion(val_outputs, labels)
        validation_loss += val_loss.item()
        _, val_preds = torch.max(val_outputs.data, 1)
        validation_correct += torch.sum(val_preds == labels.data)

        # Collect for metrics calculation
        all_preds.extend(val_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    
    validation_loss /= len(val_loader)
    print(f'Epoch {epoch+1}, Validation Loss: {validation_loss:.4f}')

    train_accuracy = train_correct.double().item() / len(train_dataset_final)
    validation_accuracy = validation_correct.double().item() / len(val_dataset_final)

    print(f'Epoch {epoch+1}, Training Accuracy: {train_accuracy:.4f}')
    print(f'Epoch {epoch+1}, Validation Accuracy: {validation_accuracy:.4f}')

    # Updating scheduler according to accuracy to try and reach maximum performance

    train_loss_history.append(train_loss)
    val_loss_history.append(validation_loss)
    train_acc_history.append(train_accuracy)
    val_acc_history.append(validation_accuracy)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    scheduler.step(validation_loss)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Classification report
    print(classification_report(all_labels, all_preds, target_names=label_mapping.keys()))


    # Save the model if it has the best validation accuracy to avoid overfitting
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        torch.save(model.state_dict(), 'model/deep_emotion.pth')
        print("Saved new best model")
        conf_mat = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=label_mapping.keys(),yticklabels=label_mapping.keys())
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png')

    # Implementing early stopping logic to save computational resources in case of overfitting
    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print(f"No improvement in validation loss for {epochs_without_improvement} epochs")

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break  # Exit from the training loop

train_loss_history = np.array(train_loss_history)
val_loss_history = np.array(val_loss_history)
train_acc_history = np.array(train_acc_history)
val_acc_history = np.array(val_acc_history)

generate_plot(train_loss_history, val_loss_history, train_acc_history, val_acc_history, label="Emotion Recognition CNN")
