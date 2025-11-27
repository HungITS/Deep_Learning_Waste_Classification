import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import ResNet50
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 100
WARMUP_EPOCHS = 5
NUM_CLASSES = 6
DATA_TRAIN = "data/train"
DATA_TEST  = "data/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.cuda.empty_cache()

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(root=DATA_TRAIN, transform=train_transform)
test_dataset  = datasets.ImageFolder(root=DATA_TEST,  transform=test_transform)

labels = [s[1] for s in train_dataset.samples]
class_count = Counter(labels)
weights = torch.tensor([1.0 / class_count[i] for i in range(NUM_CLASSES)], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

model = ResNet50(num_classes=NUM_CLASSES).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

def warmup_lr(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    return 1.0

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

best_acc = 0.0
patience_counter = 0
max_patience = 15

# Tracking loss và accuracy
train_losses = []
test_accuracies = []
Path("images").mkdir(exist_ok=True) 

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()

    if epoch < WARMUP_EPOCHS:
        warmup_scheduler.step()

    # === Evaluation ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Lưu loss và accuracy
    train_losses.append(avg_loss)
    test_accuracies.append(acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}% | LR: {current_lr:.6f}")

    if epoch >= WARMUP_EPOCHS:
        scheduler.step(acc)

    if acc > best_acc:
        best_acc = acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_resnet50.pth")
        print(f"New best model saved! Accuracy: {acc:.2f}%")
    else:
        patience_counter += 1
        
    if patience_counter >= max_patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break

print(f"\nBest Accuracy: {best_acc:.2f}%")

# Vẽ biểu đồ loss và accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, 'r-', label='Test Accuracy')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('images/training_progress.png', dpi=300, bbox_inches='tight')
print("\n📈 Biểu đồ training đã lưu vào: images/training_progress.png")
plt.close()


