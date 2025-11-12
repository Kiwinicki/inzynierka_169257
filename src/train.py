import torch
import torch.nn as nn
from .models import PlainCNN, ResNet
from .data_loaders import get_data_loaders
from pathlib import Path
from src.dataset import CLASS_LABELS
import argparse
import math
from torch.utils.tensorboard import SummaryWriter


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
        self.global_step = 0
        self.run_name = f"{args.arch}-{args.base_ch}ch-{args.lr:.2e}lr"
        self.writer = SummaryWriter(Path("runs") / self.run_name)

    def train_epoch(self):
        self.model.train()
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.opt.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(torch.log_softmax(outputs, dim=1), labels)
            loss.backward()
            self.opt.step()

            self.global_step += 1
            if self.global_step % 100 == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)

        self.writer.flush()

    def valid_epoch(self, curr_ep):
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
        valid_loss /= len(self.valid_loader)

        print(f"Valid epoch {curr_ep} acc: {valid_acc:.2f}, loss: {valid_loss:.3f}")

        self.writer.add_scalar("valid/acc", valid_acc, self.global_step)
        self.writer.add_scalar("valid/loss", valid_loss, self.global_step)
        self.writer.flush()
        return valid_loss

    def stop_early(self, valid_loss, curr_ep):
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
            print(f"Validation loss isn't improving, stopping early at {curr_ep} epoch")
            return True

    def train(self):
        curr_ep = 0
        valid_loss = math.inf
        while curr_ep < self.args.num_epochs:
            self.train_epoch()
            valid_loss = self.valid_epoch(curr_ep)
            if self.stop_early(valid_loss, curr_ep):
                break
            curr_ep += 1

        self.writer.add_hparams(
            {
                "num_epochs": curr_ep,
                "batch_size": self.args.batch_size,
                "lr": self.args.lr,
                "base_ch": self.args.base_ch,
                "arch": self.args.arch,
            },
            {"final/valid_loss": valid_loss},
            run_name="final",
        )
        self.writer.flush()
        self.writer.close()

        Path("./checkpoints/").mkdir(exist_ok=True)
        torch.save(
            self.model.state_dict(), f"./checkpoints/{self.args.arch}-{curr_ep}ep.ckpt"
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
