import yaml
import subprocess
from pathlib import Path


def run_experiment(name, config):
    print(f"=== Starting experiment: {name} ===")

    cmd = ["python", "-m", "src.train"]

    for key, value in config.items():
        if isinstance(value, list):
            cmd.append(f"--{key}")
            cmd.extend([str(v) for v in value])
        elif isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    cmd.extend(["--run_name", name])

    print(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"=== Experiment {name} completed successfully ===\n")
    except subprocess.CalledProcessError as e:
        print(f"!!! Experiment {name} failed with code {e.returncode} !!!\n")


def main():
    config_path = Path("src/train_configs.yml")
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    for name, config in configs.items():
        run_experiment(name, config)


if __name__ == "__main__":
    main()
