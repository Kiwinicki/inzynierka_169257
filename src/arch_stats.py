import torch
from tabulate import tabulate
from src.models import ARCHITECTURES


def main():
    results = []
    print(
        f"Benchmarking architectures on {'CUDA' if torch.cuda.is_available() else 'CPU'}..."
    )

    for name, model_cls in ARCHITECTURES.items():
        try:
            model = model_cls(base_ch=32, num_classes=8)

            # 1. Parameter Count
            params = model.num_parameters
            params_m = params / 1e6

            # 2. FLOPs
            try:
                flops_count = model.flops()
                flops_g = flops_count / 1e9
            except Exception as e:
                flops_g = "N/A"

            # 3. Peak Memory
            try:
                mem_bytes = model.activation_memory_bytes(is_training=False)
                mem_mb = mem_bytes / (1024 * 1024)
            except Exception as e:
                mem_mb = "N/A"

            results.append(
                [
                    name,
                    f"{params_m:.2f} M",
                    f"{flops_g:.2f} G" if isinstance(flops_g, float) else flops_g,
                    f"{mem_mb:.2f} MB" if isinstance(mem_mb, float) else mem_mb,
                ]
            )

        except Exception as e:
            results.append([name, "Error", "Error", f"Error: {str(e)}"])

    print(
        tabulate(
            results,
            headers=["Architecture", "Params", "FLOPs", "Peak Mem (Activations)"],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
