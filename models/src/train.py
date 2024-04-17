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
import argparse

import warnings
warnings.filterwarnings('ignore')

import os
from torchvision import transforms, datasets, utils
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import importlib
from custom_dataset import CustomDataset
from collections import Counter

def Train(model, epochs, train_loader, val_loader, criterion, scheduler, optimizer, device, label_mapping):
    pati = 40
    epochs_without_improvement = 0
    best_accuracy = 0.0
    best_val_loss = np.inf
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    print("===================================Training Starting===================================")
    for epoch in range(epochs):
        # Set predictions and labels lists
        print(f"Starting {epoch+1} epoch")
        all_preds = [] 
        all_labels = []
        # Training loop
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
        scheduler.step(validation_accuracy)

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

        if epochs_without_improvement >= pati:
            print("Early stopping triggered.")
            break  # Exit from the training loop

        print("===================================Training Finished===================================")

    train_loss_history = np.array(train_loss_history)
    val_loss_history = np.array(val_loss_history)
    train_acc_history = np.array(train_acc_history)
    val_acc_history = np.array(val_acc_history)

    generate_plot(train_loss_history, val_loss_history, train_acc_history, val_acc_history, label="Emotion Recognition CNN")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration of setup and training process')
    parser.add_argument('-d', '--data', type=str, required=True, help='Data folder that contains data files')
    parser.add_argument('--hyperparams', type=bool, default=False, help='True when changing the hyperparameters e.g (batch size, LR, num. of epochs)')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Value of learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Training/Validation batch size')
    parser.add_argument('-t', '--train', type=bool, default=False, help='True when training')
    args = parser.parse_args()

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size

    if args.train:
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')

        # Define transformations
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.2, contrast=0.2)], p=0.3),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.1, 0.1))], p=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])


        # Define the label mapping
        label_mapping = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 'neutral': 5, 'sad': 6, 'surprise': 7}
        #class_difficulty = {0: 1.3, 1: 1.2, 2: 1.9, 3: 0.78, 4: 1.1, 5: 1.12, 6: 0.98}

        # Initialize datasets
        train_dataset_final = CustomDataset(f'{args.data}/train/', label_mapping=label_mapping, transform=train_transform, balance_dataset=True)
        val_dataset_final = CustomDataset(f'{args.data}/validation/', label_mapping=label_mapping, transform=transform)
        print(f"Training samples: {len(train_dataset_final)}")
        print(f"Class distribution: {train_dataset_final.get_class_distribution()}")
        print(f"Validation samples: {len(val_dataset_final)}")

        # Calculate class weights for loss function
        total_samples = len(train_dataset_final)
        class_weights = {class_id : total_samples/ ( 8 * dict(train_dataset_final.get_class_distribution())[class_id] ) for class_id in dict(train_dataset_final.get_class_distribution()).keys()}
        weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float)

        # Create the data loaders
        train_loader = DataLoader(train_dataset_final, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset_final, batch_size=batch_size, shuffle=False, num_workers=2)

        # Initialize the model, optimizer, and loss function
        deep_emotion = Deep_Emotion().to(device)
        #model = MobileNet().to(device)
        optimizer = optim.SGD(deep_emotion.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9, nesterov=True)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))  # Cross-entropy loss
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.75, patience=7, verbose=True, min_lr=1e-6)

        # Train the model
        Train(deep_emotion, epochs, train_loader, val_loader, criterion, scheduler, optimizer, device, label_mapping)