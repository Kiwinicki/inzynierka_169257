import math
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
        self.valid_acc = None
        self.should_stop = False


class Hook:
    def on_train_begin(self, trainer, state):
        pass

    def on_train_end(self, trainer, state):
        pass

    def on_train_step_end(self, trainer, state, loss):
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
            path = (
                self.save_dir
                / f"{trainer.args.arch}-ep{state.epoch}-loss{state.valid_loss:.3f}.ckpt"
            )

            checkpoint = {
                "state_dict": trainer.model.state_dict(),
                "arch": trainer.args.arch,
                "base_ch": trainer.args.base_ch,
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

    def on_train_begin(self, trainer, state):
        self.register_activation_hooks(trainer.model)

    def on_train_step_end(self, trainer, state, loss):
        self.current_step = state.global_step
        if state.global_step % 100 == 0:
            self.writer.add_scalar("train/loss", loss.item(), state.global_step)
            self._log_weights(trainer.model, state.global_step)

    def on_train_epoch_end(self, trainer, state):
        self.writer.flush()

    def on_valid_epoch_end(self, trainer, state):
        self.writer.add_scalar("valid/acc", state.valid_acc, state.global_step)
        self.writer.add_scalar("valid/loss", state.valid_loss, state.global_step)
        self.writer.flush()

    def on_train_end(self, trainer, state):
        self.writer.add_hparams(
            {
                "num_epochs": state.epoch,
                "batch_size": trainer.args.batch_size,
                "lr": trainer.args.lr,
                "base_ch": trainer.args.base_ch,
                "arch": trainer.args.arch,
            },
            {"final/valid_loss": state.valid_loss, "final/valid_acc": state.valid_acc},
            run_name="final",
        )
        self.writer.flush()

    def _log_weights(self, model, step):
        for name, param in model.named_parameters():
            if "weight" in name:
                self.writer.add_histogram(f"weights/{name}", param, step)

    def close(self):
        for hook in self.activation_hooks:
            hook.remove()
        self.writer.close()
