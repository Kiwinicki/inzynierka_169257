import torch
import torch.nn as nn
import trackio
from .models import PlainCNN, ResNet
from .data_loaders import get_data_loaders
from pathlib import Path
from src.dataset import CLASS_LABELS
import argparse
import math


ARCHITECTURES = {"plain": PlainCNN, "resnet": ResNet}


class Trainer:
    def __init__(self, args):
        self.args = args
        model_cls = ARCHITECTURES[args.arch]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_cls(base_ch=args.base_ch, n_classes=len(CLASS_LABELS)).to(
            self.device
        )
        self.train_loader, self.valid_loader, _ = get_data_loaders(
            batch_size=args.batch_size
        )
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.best_loss = math.inf
        self.num_bad = 0

        trackio.init(
            project="emotion-recognition",
            name=f"{args.arch}-{args.base_ch}ch-{args.lr}lr-{args.batch_size}bs",
            config={
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "base_ch": args.base_ch,
                "arch": args.arch,
            },
        )

    def train_epoch(self):
        self.model.train()
        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.opt.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(torch.log_softmax(outputs, dim=1), labels)
            loss.backward()
            self.opt.step()
            if i % 100 == 0:
                print("loss:", loss.item())
                trackio.log({"train_loss": loss.item()})

    def valid_epoch(self):
        self.model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(torch.log_softmax(outputs, dim=1), labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                true_labels = torch.argmax(labels, dim=1)
                total += labels.size(0)
                correct += (predicted == true_labels).sum().item()

        valid_acc = 100 * correct / total
        print("Validation acc:", valid_acc)
        valid_loss /= len(self.valid_loader)
        print("Validation loss:", valid_loss)
        trackio.log({"valid_acc": valid_acc, "valid_loss": valid_loss})
        return valid_loss

    def stop_early(self, valid_loss):
        if self.best_loss == math.inf:
            self.best_loss = valid_loss
            return False
        elif valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.num_bad = 0
            return False
        elif self.num_bad < 5 and valid_loss >= self.best_loss:
            self.num_bad += 1
            return False
        else:
            print("Validation loss isn't improving, stopping early")
            return True

    def train(self):
        curr_ep = 0
        while curr_ep < self.args.num_epochs:
            self.train_epoch()
            valid_loss = self.valid_epoch()
            if self.stop_early(valid_loss):
                break
            curr_ep += 1

        trackio.finish()
        Path("./checkpoints/").mkdir(exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"./checkpoints/{self.args.arch}-{curr_ep}ep.ckpt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument(
        "--arch", type=str, choices=ARCHITECTURES.keys(), default="plain"
    )
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
