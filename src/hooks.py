import math
import time
import torch
from pathlib import Path
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TrainerState:
    def __init__(self):
        self.global_step = 0
        self.epoch = 0
        self.train_loss = None
        self.valid_loss = None
        self.train_acc = None
        self.valid_acc = None
        self.should_stop = False


class Hook:
    def on_train_begin(self, trainer, state):
        pass

    def on_train_epoch_begin(self, trainer, state):
        pass

    def on_train_end(self, trainer, state):
        pass

    def on_train_step_end(self, trainer, state, **metrics):
        pass

    def on_train_epoch_end(self, trainer, state):
        pass

    def on_valid_epoch_end(self, trainer, state):
        pass


class EarlyStoppingHook(Hook):
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = math.inf
        self.num_bad = 0

    def on_valid_epoch_end(self, trainer, state):
        if state.valid_loss < self.best_loss:
            self.best_loss = state.valid_loss
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                print(
                    f"Validation loss isn't improving, stopping early at epoch {state.epoch}"
                )
                state.should_stop = True


class CheckpointHook(Hook):
    def __init__(self, save_dir="./checkpoints", save_best_only=True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.save_best_only = save_best_only
        self.best_loss = math.inf
        self.best_path = None

    def on_valid_epoch_end(self, trainer, state):
        if not self.save_best_only or state.valid_loss < self.best_loss:
            self.best_loss = state.valid_loss
            path = self.save_dir / f"{trainer.args.run_name}.ckpt"

            checkpoint = {
                "state_dict": trainer.model.state_dict(),
                "args": vars(trainer.args),
                "epoch": state.epoch,
                "loss": state.valid_loss,
            }

            if (
                self.save_best_only
                and self.best_path is not None
                and self.best_path.exists()
            ):
                self.best_path.unlink()

            torch.save(checkpoint, path)
            self.best_path = path


class LoggerHook(Hook):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.activation_hooks = []
        self.current_step = None
        self.train_start_time = None
        self.epoch_start_time = None
        self.epoch_durations = []

    def register_activation_hooks(self, model):
        def get_activation_hook(name):
            def hook(model, input, output):
                if self.current_step is not None and self.current_step % 100 == 0:
                    self.writer.add_histogram(
                        f"activations/{name}", output, self.current_step
                    )

            return hook

        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                self.activation_hooks.append(
                    layer.register_forward_hook(get_activation_hook(name))
                )

    def on_train_begin(self, trainer, state, **metrics):
        self.train_start_time = time.time()
        self.register_activation_hooks(trainer.model)
        hparams = vars(trainer.args).copy()
        for k, v in hparams.items():
            if isinstance(v, (list, tuple)):
                hparams[k] = str(v)

        self.writer.add_hparams(
            hparams,
            metric_dict={},
            run_name=".",
        )

    def on_train_epoch_begin(self, trainer, state, **metrics):
        self.epoch_start_time = time.time()

    def on_train_step_end(self, trainer, state, **metrics):
        self.current_step = state.global_step
        if state.global_step % 100 == 0:
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.writer.add_scalar(f"train/{name}", value, state.global_step)
            self._log_weights(trainer.model, state.global_step)
            self.writer.add_scalar(
                "train/lr", trainer.opt.param_groups[0]["lr"], state.global_step
            )

    def on_train_epoch_end(self, trainer, state, **metrics):
        if self.epoch_start_time is not None:
            duration = time.time() - self.epoch_start_time
            self.epoch_durations.append(duration)
            self.writer.add_scalar("time/train_epoch", duration, state.epoch)
        self.writer.flush()

    def on_valid_epoch_end(self, trainer, state, **metrics):
        self.writer.add_scalar("valid/acc", state.valid_acc, state.global_step)
        self.writer.add_scalar("valid/loss", state.valid_loss, state.global_step)
        self.writer.flush()

    def _log_weights(self, model, step):
        for name, param in model.named_parameters():
            if "weight" in name:
                self.writer.add_histogram(f"weights/{name}", param, step)

    def close(self):
        for hook in self.activation_hooks:
            hook.remove()
        self.writer.close()

    def on_train_end(self, trainer, state):
        if self.train_start_time is not None:
            total_time = time.time() - self.train_start_time
            self.writer.add_scalar(
                "time/total_train_time", total_time, state.global_step
            )

        if self.epoch_durations:
            mean_duration = sum(self.epoch_durations) / len(self.epoch_durations)
            self.writer.add_scalar(
                "time/mean_epoch_duration", mean_duration, state.global_step
            )


class LinearWarmupHook(Hook):
    def __init__(self, start_lr=None, end_lr=None, num_epochs=None):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_epochs = int(num_epochs)

    def on_train_begin(self, trainer, state, **metrics):
        if self.start_lr is None:
            self.start_lr = trainer.args.lr * 0.1
        if self.end_lr is None:
            self.end_lr = trainer.args.lr
        if self.num_epochs is None:
            self.num_epochs = min(int(trainer.args.num_epochs * 0.1), 10)

        dataset_len = len(trainer.train_loader.dataset)
        self.n_warmup_steps = (dataset_len // trainer.args.batch_size) * self.num_epochs

        def lr_lambda(step):
            if step >= self.n_warmup_steps:
                return self.end_lr / trainer.args.lr
            return (
                self.start_lr
                + (self.end_lr - self.start_lr) * step / self.n_warmup_steps
            ) / self.end_lr

        self.sched = torch.optim.lr_scheduler.LambdaLR(trainer.opt, lr_lambda=lr_lambda)

    def on_train_step_end(self, trainer, state, **metrics):
        self.sched.step()
