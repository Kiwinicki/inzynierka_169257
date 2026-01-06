import torch
import torch.nn as nn
from .models import ARCHITECTURES
from .data_loaders import get_data_loaders
from src.dataset import CLASS_LABELS
from .hooks import TrainerState
from src.metrics import Metrics, TieAwareAccuracy


class Trainer:
    def __init__(self, args, hooks=None, model=None):
        self.args = args
        self.hooks = hooks or []
        self.state = TrainerState()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        if model is not None:
            self.model = model.to(self.device)
        else:
            model_cls = ARCHITECTURES[args.arch]
            kwargs = {"base_ch": args.base_ch, "num_classes": len(CLASS_LABELS)}
            if args.stages:
                kwargs["stages"] = args.stages
            self.model = model_cls(**kwargs).to(self.device)

        self.train_loader, self.valid_loader, self.test_loader = get_data_loaders(
            batch_size=args.batch_size, oversample=getattr(args, "oversample", False)
        )

        if getattr(args, "class_weighting", False):
            self.criterion = self._weighted_kl_div
        else:
            self.criterion = nn.KLDivLoss(reduction="batchmean")

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=0.05
        )
        self.metrics = Metrics(num_classes=len(CLASS_LABELS), device=self.device)
        self.train_acc = TieAwareAccuracy().to(self.device)
        self.valid_acc = TieAwareAccuracy().to(self.device)
        self.class_weights = self.train_loader.dataset.get_class_weights().to(
            self.device
        )

        self.use_amp = getattr(args, "use_amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _call_hook(self, event, **kwargs):
        for hook in self.hooks:
            getattr(hook, event)(self, self.state, **kwargs)

    def _weighted_kl_div(self, log_q, labels):
        argmax_class = torch.argmax(labels, dim=1)
        sample_w = self.class_weights[argmax_class]
        kl_per_sample = torch.nn.functional.kl_div(log_q, labels, reduction="none").sum(
            dim=1
        )
        return (sample_w * kl_per_sample).mean()

    def train_epoch(self):
        self._call_hook("on_train_epoch_begin")
        self.model.train()
        self.train_acc.reset()
        total_loss = 0.0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.opt.zero_grad()

            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                outputs = self.model(images)
                log_q = torch.log_softmax(outputs, dim=1)
                loss = self.criterion(log_q, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            total_loss += loss.item()
            acc = self.train_acc(outputs, labels) * 100

            self.state.global_step += 1
            self._call_hook("on_train_step_end", loss=loss, acc=acc)

        self.state.train_acc = self.train_acc.compute() * 100
        self.state.train_loss = total_loss / len(self.train_loader)
        self._call_hook("on_train_epoch_end")

    def valid_epoch(self, curr_ep):
        self.model.eval()
        self.valid_acc.reset()
        valid_loss = 0.0

        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    outputs = self.model(images)
                    loss = self.criterion(torch.log_softmax(outputs, dim=1), labels)

                valid_loss += loss.item()
                self.valid_acc.update(outputs, labels)

        self.state.valid_acc = self.valid_acc.compute() * 100
        self.state.valid_loss = valid_loss / len(self.valid_loader)
        self.state.epoch = curr_ep

        print(
            f"epoch {curr_ep} Valid acc: {self.state.valid_acc:.2f}, loss: {self.state.valid_loss:.3f} | Train acc: {self.state.train_acc:.2f}, loss: {self.state.train_loss:.3f}"
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

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    outputs = self.model(images)

                self.metrics.update(outputs, labels)

        results = self.metrics.compute()
        return results
