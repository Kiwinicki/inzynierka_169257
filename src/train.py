import argparse
from src.models import ARCHITECTURES
from src.trainer import Trainer
from src.hooks import EarlyStoppingHook, CheckpointHook, LoggerHook, LinearWarmupHook
from pathlib import Path
from datetime import datetime
from src.dataset import CLASS_LABELS
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")  # make plotting more robust (not dependent on GUI)
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def create_conf_matrix(conf_mat):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(conf_mat, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(conf_mat.shape[1]),
        yticks=np.arange(conf_mat.shape[0]),
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = conf_mat.max() / 2.0
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(
                j,
                i,
                format(conf_mat[i, j], ".0f"),
                ha="center",
                va="center",
                color="white"
                if conf_mat[i, j] > thresh
                else "black",  # For readability
            )
    fig.tight_layout()
    return fig


def evaluate_model(trainer, logger_hook, args):
    print("Starting evaluation...")
    writer = logger_hook.writer

    results = trainer.test()

    conf_mat = results["conf_mat"].cpu().numpy()
    fig = create_conf_matrix(conf_mat)
    writer.add_figure(
        "test/confusion_matrix", fig, global_step=trainer.state.global_step
    )
    plt.close(fig)

    # get only scalar metrics
    metrics = {k: v.item() for k, v in results.items() if v.numel() == 1}

    for name, value in metrics.items():
        writer.add_scalar(f"test/{name}", value, global_step=trainer.state.global_step)

    writer.flush()
    print(f"Evaluation results logged to {writer.log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument(
        "--arch", type=str, choices=ARCHITECTURES.keys(), default="plain"
    )
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--class_weighting", action="store_true")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--stages", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{args.arch}-{args.base_ch}ch-{args.lr:.2e}lr"
        for arg, value in sorted(vars(args).items()):  # add boolean flags to run name
            if isinstance(value, bool) and value:
                run_name += f"-{arg}"
        run_name += f"-{datetime.now().strftime('%y%m%d-%H:%M')}"
    args.run_name = run_name

    logger_hook = LoggerHook(Path("runs") / run_name)
    # early_stopping_hook = EarlyStoppingHook(patience=5)
    checkpoint_hook = CheckpointHook(save_dir="./checkpoints")
    warmup_hook = LinearWarmupHook(num_epochs=args.warmup_epochs)
    hooks = [logger_hook, checkpoint_hook, warmup_hook]  # , early_stopping_hook

    trainer = Trainer(args, hooks=hooks)

    try:
        trainer.train()
        evaluate_model(trainer, logger_hook, args)
    finally:
        logger_hook.close()
