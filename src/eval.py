import argparse
import torch
from src.trainer import Trainer
from src.models.base import BaseCNN
from src.dataset import CLASS_LABELS

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    model, loaded_args = BaseCNN.from_checkpoint(
        args.checkpoint,
        num_classes=len(CLASS_LABELS)
    )
    
    args.arch = loaded_args["arch"]
    args.base_ch = loaded_args["base_ch"]
    args.lr = 0.0 
    args.num_epochs = 0
    
    trainer = Trainer(args, model=model)
    results = trainer.test()
    for metric_name, value in results.items():
        if value.numel() == 1:
            print(f"{metric_name}: {value.item():.4f}")
        else:
            print(f"{metric_name}:\n{value.cpu().numpy()}")
