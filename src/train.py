import torch
import torch.nn as nn
import torch.optim as optim
import trackio
from .models import PlainCNN, ResNet
from .data_loaders import get_data_loaders
from pathlib import Path
from src.dataset import CLASS_LABELS
import argparse

ARCHITECTURES = {"plain": PlainCNN, "resnet": ResNet}


parser = argparse.ArgumentParser("Training")
parser.add_argument("num_epochs", type=int, default=1)
parser.add_argument("batch_size", type=int, default=64)
parser.add_argument("lr", type=float, default=1e-3)
parser.add_argument("base_ch", type=int, default=32)
parser.add_argument(
    "arch", type=str, choices=ARCHITECTURES.keys(), default=ARCHITECTURES["plain"]
)
args = parser.parse_args()

model_cls = ARCHITECTURES[args.arch]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_cls(base_ch=args.base_ch, n_classes=len(CLASS_LABELS)).to(device)
train_loader, val_loader, _ = get_data_loaders(batch_size=args.bs)
criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=args.lr)


trackio.init(
    project="emotion-recognition",
    name=f"{args.arch}-{args.base_ch}ch-{args.lr}-{args.batch_size}bs",
    config={
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "base_ch": args.base_ch,
        "arch": args.arch,
    },
)

for epoch in range(args.epochs):
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
            trackio.log({"train_loss": loss.item()})

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
    trackio.log({"val_acc": val_acc, "val_loss": val_loss})


Path("./checkpoints/").mkdir(exist_ok=True)
torch.save(
    model.state_dict(), f"./checkpoints/{model.__class__.__name__}-{args.epochs}ep.ckpt"
)
trackio.finish()
