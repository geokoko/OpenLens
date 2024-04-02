import torch
import torch.optim as optim
from emotion_cnn import EmotionCNN
from sklearn.metrics import f1_score

model = EmotionCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss() # Cross-entropy loss for classification problems
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
best_accuracy = 0.0

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device) # transfer image data and labels to GPU
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_dataloader)}')
    scheduler.step()
    
    # Set predictions and labels lists
    all_preds = [] 
    all_labels = []

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device) # tranfer image data and labels to GPU

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect for F1 calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    accuracy = 100 * correct / total
    print(f'Accuracy on the validation set: {accuracy}%')

    f1 = f1_score(all_labels, all_preds, average='weighted')  # 'weighted' accounts for label imbalance
    print(f'F1 Score on the validation set: {f1}')

    # Save the model if it has the best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved new best model")