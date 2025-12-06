import argparse
from src.models import ARCHITECTURES
from src.trainer import Trainer
from src.hooks import EarlyStoppingHook, CheckpointHook, LoggerHook
from pathlib import Path


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

    run_name = f"{args.arch}-{args.base_ch}ch-{args.lr:.2e}lr"

    # Initialize hooks
    logger_hook = LoggerHook(Path("runs") / run_name)
    early_stopping_hook = EarlyStoppingHook(patience=5)
    checkpoint_hook = CheckpointHook(save_dir="./checkpoints")
    hooks = [logger_hook, early_stopping_hook, checkpoint_hook]

    trainer = Trainer(args, hooks=hooks)

    try:
        trainer.train()
    finally:
        logger_hook.close()
