import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.deep_emotion import Deep_Emotion
from src.models.mobilenet import MobileNet
from src.models.resnet import ResNet
from src.utils.custom_dataset import CustomDataset
from src.utils.generate_plot import generate_plot

def train_model(model, epochs, train_loader, val_loader, criterion, optimizer, scheduler, device, label_mapping, save_path):
    best_accuracy = 0.0
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    print(f"Training on device: {device}")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.item())

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if scheduler:
            scheduler.step(val_acc)

        # Deep Copy Model if best
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")

    # Final Metrics & Plotting
    print("Training Complete.")
    print(f"Best Val Acc: {best_accuracy:.4f}")

    # Generate Plots
    generate_plot(np.array(train_loss_history), np.array(val_loss_history), 
                  np.array(train_acc_history), np.array(val_acc_history), 
                  label="Training_Log")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Models')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root (containing train/ and validation/ folders)')
    parser.add_argument('--model', type=str, default='deep_emotion', choices=['deep_emotion', 'mobilenet', 'resnet'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save models')

    args = parser.parse_args()

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Setup Directories
    os.makedirs(args.save_dir, exist_ok=True)

    # 3. Label Mapping
    label_mapping = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    num_classes = len(label_mapping)

    # 4. Transforms
    # DeepEmotion expects 48x48. MobileNet/ResNet typically 224x224    
    # Since we modified MobileNet/ResNet first layers to take 1 channel, we can stick to 48x48 
    # OR upscale to 224x224 for potentially better transfer learning feature extraction.
    # stick to 48x48 for speed on Pi, unless it proves too small for ResNet features.
    input_size = 48 

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Grayscale normalization
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 5. Datasets
    print("Loading Datasets...")
    train_dataset = CustomDataset(os.path.join(args.data, 'train'), label_mapping, train_transform, balance_dataset=True)
    val_dataset = CustomDataset(os.path.join(args.data, 'test'), label_mapping, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    # 6. Model Init
    print(f"Initializing {args.model}...")
    if args.model == 'deep_emotion':
        model = Deep_Emotion(num_classes=num_classes)
    elif args.model == 'mobilenet':
        model = MobileNet(num_classes=num_classes, pretrained=True)
    elif args.model == 'resnet':
        model = ResNet(num_classes=num_classes, pretrained=True)

    model = model.to(device)

    # 7. Optimizer & Loss
    # Calculate class weights
    class_counts = train_dataset.get_class_distribution()
    total_samples = len(train_dataset)
    # Avoid div by zero
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        if count > 0:
            weights.append(total_samples / (num_classes * count))
        else:
            weights.append(1.0)

    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 8. Run Training
    save_path = os.path.join(args.save_dir, f'{args.model}_best.pt')
    train_model(model, args.epochs, train_loader, val_loader, criterion, optimizer, scheduler, device, label_mapping, save_path)
