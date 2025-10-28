import torch
import torch.nn as nn
import torch.optim as optim
import trackio as wandb
from models import PlainCNN, ResNet
from .data_loaders import get_data_loaders
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlainCNN(base_ch=32, n_classes=8).to(device)
EPOCHS = 20
train_loader, val_loader, _ = get_data_loaders(batch_size=64)
criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(torch.log_softmax(outputs, dim=1), labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("loss:", loss.item())

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(torch.log_softmax(outputs, dim=1), labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            true_labels = torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()

    val_acc = 100 * correct / total
    print("Validation acc:", val_acc)
    val_loss /= len(val_loader)
    print("Validation loss:", val_loss)


Path("../checkpoints/").mkdir(exist_ok=True)
torch.save(
    model.state_dict(), f"../checkpoints/{model.__class__.__name__}-{EPOCHS}ep.ckpt"
)
