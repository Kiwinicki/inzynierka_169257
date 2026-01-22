import argparse
import json
import torch
import glob
from pathlib import Path
from src.trainer import Trainer
from src.models.base import BaseCNN
from src.dataset import CLASS_LABELS


def convert_tensor(obj):
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.cpu().tolist()
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation")
    parser.add_argument(
        "checkpoints", type=str, nargs="+", help="Paths to checkpoint files (supports glob)"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    checkpoint_files = []
    for pattern in args.checkpoints:
        matches = glob.glob(pattern)
        if not matches:
            checkpoint_files.append(pattern)
        else:
            checkpoint_files.extend(matches)
            
    checkpoint_files = sorted(list(set(checkpoint_files)))
    
    if not checkpoint_files:
        print("No checkpoints found!")
        exit(1)

    print(f"Found {len(checkpoint_files)} checkpoints to evaluate.")

    results_dict = {}

    for ckpt_path in checkpoint_files:
        print(f"\nEvaluating: {ckpt_path}")
        try:
            model, loaded_args = BaseCNN.from_checkpoint(
                ckpt_path, num_classes=len(CLASS_LABELS)
            )
        except Exception as e:
            print(f"Failed to load {ckpt_path}: {e}")
            continue

        eval_args = argparse.Namespace()
        eval_args.batch_size = args.batch_size
        eval_args.arch = loaded_args.get("arch", "unknown")
        eval_args.base_ch = loaded_args.get("base_ch", 64) 
        eval_args.stages = loaded_args.get("stages", None)
        eval_args.lr = 0.0
        eval_args.num_epochs = 0
        eval_args.oversample = False
        eval_args.class_weighting = False
        eval_args.use_amp = True

        try:
            if torch.cuda.is_available():
                model = model.cuda()
            
            flops = model.flops()
            
            gflops = flops / 1e9
            
            act_mem_bytes = model.activation_memory_bytes(is_training=False)
            act_mem_mb = act_mem_bytes / (1024 * 1024)
            
            model_size_bytes = model.model_size_bytes()
            model_size_mb = model_size_bytes / (1024 * 1024)
            
            params = model.num_parameters
            
            print(f"  > Params: {params/1e6:.2f}M")
            print(f"  > GFLOPs: {gflops:.2f}")
            print(f"  > Peak Activation Mem: {act_mem_mb:.2f} MB")
            
        except Exception as e:
            print(f"Warning: Failed to calculate complexity metrics: {e}")
            gflops = 0.0
            act_mem_mb = 0.0
            model_size_mb = 0.0
            params = 0

        trainer = Trainer(eval_args, model=model)
        metrics = trainer.test()
        clean_metrics = {k: convert_tensor(v) for k, v in metrics.items()}
        
        clean_metrics["gflops"] = gflops
        clean_metrics["activation_mem_mb"] = act_mem_mb
        clean_metrics["model_size_mb"] = model_size_mb
        clean_metrics["params_m"] = params / 1e6
        
        ckpt_name = Path(ckpt_path).stem
        results_dict[ckpt_name] = clean_metrics
        print(f"Finished {ckpt_name}")

    output_file = "eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=4)
        
    print(f"\nSaved results to {output_file}")
