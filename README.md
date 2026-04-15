# CNN Translation Robustness: MaxPool vs. AvgPool

An ablation study exploring the impact of downsampling mechanisms and translation augmentation on the robustness of CNN classifiers using the CIFAR-10 dataset.

## Introduction
This project examines translation robustness in CNN-based image classification: how consistently a model predicts the correct label when an image is shifted by an increasing number of pixels.

## Research Question
How do a Convolutional Neural Network's downsampling choices and the strength of translation augmentation impact robustness to spatial translation in CIFAR-10?

## Hypothesis
Translation augmentation will increase robustness to spatial shifts. As augmentation becomes stronger, the performance gap between MaxPool and AvgPool will tend to decrease, suggesting that augmentation-driven invariance reduces the influence of architectural inductive bias.

## Project Structure
- `src/models.py`: CNN architecture definition.
- `src/data_loader.py`: CIFAR-10 data pipeline with augmentation.
- `src/engine.py`: Training logic.
- `src/utils.py`: Evaluation and helper functions.
- `main.py`: Entry point for running the ablation study.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ibrahim-2337/cnn_translation_robustness.git
   cd cnn_translation_robustness
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the full experiment grid (6 conditions):
```bash
python main.py
```

### What happens when you run `main.py`?
1. **Data Loading**: The CIFAR-10 dataset is automatically downloaded and processed with the appropriate translation augmentation level.
2. **Model Training**: Six different models are trained, corresponding to each experimental condition (MaxPool vs. AvgPool for 0px, 2px, and 4px augmentation).
3. **Evaluation**: After training each model, it is evaluated on a series of test sets shifted by 0 to 8 pixels.
4. **Results**: The training progress and a final "Robustness Score" (mean accuracy across all shifts) are printed for each condition.

## Results Summary
- **Augmentation** is the primary driver of translation robustness.
- **MaxPool** consistently provides a small but noticeable advantage over **AvgPool**, regardless of the augmentation level.
