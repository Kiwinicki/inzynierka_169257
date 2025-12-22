# Title

orig. Rozpoznawanie emocji na obrazach z użyciem konwolucyjnych sieci neuronowych (ang. Emotion Recognition on images using Convolutional Neural Networks).

# Description

orig. Celem pracy jest stworzenie systemu automatycznej klasyfikacji emocji na podstawie obrazów. Projekt obejmuje implementację modelu sieci neuronowej przy użyciu
bibliotek uczenia maszynowego w Pythonie. Rozwiązanie to znajdzie zastosowanie między innymi w analizie mediów społecznościowych, marketingu oraz badaniach
zachowań konsumenckich.

## Features
- Advanced data preprocessing and cleaning
- Multiple CNN architectures: Plain CNN, ResNet, ResNeXt, ConvNeXt
- Hyperparameter search and visualization
- Experiment tracking with Trackio
- Web app deployment with Gradio on Hugging Face Spaces

## Setup
1. Clone the repo
2. create virtual environment: `python -m venv .venv` (python 3.12) or `uv venv --python 3.12 --seed`
3. activate virtual environment: `source .venv/bin/activate` (Linux/MacOS) or `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install .` or `uv sync`
5. Download datasets: Run `python src/download_data.py`
6. Run dataset cleaning: Run `python src/clean_data.py`
7. Train models: Run `python -m src.train <optional_arguments>` (see `python -m src.train --help` for more info)
8. Show training logs: Run `tensorboard --logdir runs/`
9. Launch app: `gradio app.py`
10. Run unit tests: `python -m pytest`

## Project Structure
- `data/`: Datasets
- `models/`: Trained model weights
- `src/`: Source code
- `tests/`: Unit tests

## TODO

[x] Implement plain CNN and base training loop (baseline)

[x] Exploratory Data Analysis & Data cleaning

[x] GFLOPs and memory consumption estimation code of each model (training and inference)

[-] Data augmentation

[x] Logging with tensorboard

[x] Evaluate models on test set, compute confusion matrices, precision/recall/F1

[ ] Model comparison: accuracy, training/inference speed, memory efficiency

[-] Gradio demo app with image upload and prediction (check camera support)

[ ] Deploy the app on Hugging Face Spaces with requirements and model files

[ ] Final conclusions: best model, best compute/acc model etc. (how each trick contributed to final results)