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
2. Install dependencies: `pip install .`
3. Download datasets: Run `python src/download_data.py`
4. Run dataset cleaning: Run `python src/clean_data.py`
5. Train models: Run `python -m src.train`
6. Show training logs: Run `tensorboard --logdir runs/`
7. Launch app: `gradio app.py`

## Project Structure
- `data/`: Datasets
- `models/`: Trained model weights
- `src/`: Source code

## TODO

[x] Implement plain CNN and base training loop (baseline)

[x] Exploratory Data Analysis & Data cleaning

[x] GFLOPs and memory consumption estimation code of each model (training and inference)

[-] Data augmentation

[x] Logging with tensorboard

[ ] Evaluate models on test set, compute confusion matrices, precision/recall/F1

[ ] Model comparison: accuracy, training/inference speed, memory efficiency

[-] Gradio demo app with image upload and prediction

[ ] Deploy the app on Hugging Face Spaces with requirements and model files

[ ] Final conclusions: best model, best compute/acc model etc. (how each trick contributed to final results)

[ ] Other stuff: plot overfitting point of each model (in epochs), **justify each decision**, check other optimizers (SGD, Adam, AdamW), early stopping?, lr search, lr scheduler
