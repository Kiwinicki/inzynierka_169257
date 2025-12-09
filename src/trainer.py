import torch
import torch.nn as nn
from .models import ARCHITECTURES
from .data_loaders import get_data_loaders
from src.dataset import CLASS_LABELS
from .hooks import TrainerState
from src.metrics import Metrics


class Trainer:
    def __init__(self, args, hooks=None):
        self.args = args
        self.hooks = hooks or []
        self.state = TrainerState()

        model_cls = ARCHITECTURES[args.arch]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_cls(base_ch=args.base_ch, n_classes=len(CLASS_LABELS)).to(
            self.device
        )
        self.train_loader, self.valid_loader, self.test_loader = get_data_loaders(
            batch_size=args.batch_size
        )
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.metrics = Metrics(num_classes=len(CLASS_LABELS), device=self.device)

    def _call_hook(self, event, **kwargs):
        for hook in self.hooks:
            getattr(hook, event)(self, self.state, **kwargs)

    def train_epoch(self):
        self.model.train()
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.opt.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(torch.log_softmax(outputs, dim=1), labels)
            loss.backward()
            self.opt.step()

            self.state.global_step += 1
            self._call_hook("on_train_step_end", loss=loss)

        self._call_hook("on_train_epoch_end")

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

        self.state.valid_acc = 100 * correct / total
        self.state.valid_loss = valid_loss / len(self.valid_loader)
        self.state.epoch = curr_ep

        print(
            f"Valid epoch {curr_ep} acc: {self.state.valid_acc:.2f}, loss: {self.state.valid_loss:.3f}"
        )

        self._call_hook("on_valid_epoch_end")
        return self.state.valid_loss

    def train(self):
        self._call_hook("on_train_begin")
        curr_ep = 0
        while curr_ep < self.args.num_epochs:
            self.train_epoch()
            self.valid_epoch(curr_ep)
            if self.state.should_stop:
                break
            curr_ep += 1
        self._call_hook("on_train_end")

    def test(self):
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                self.metrics.update(outputs, labels)

        results = self.metrics.compute()
        return results
