# CNN Translation Robustness: MaxPool vs. AvgPool

An ablation study exploring the impact of downsampling mechanisms and translation augmentation on CNN classifier robustness.

## Introduction
How does a model's architecture versus its training data contribute to translation robustness? This project uses CIFAR-10 to compare `MaxPool` and `AvgPool` downsampling, with varying levels of translation-based augmentation (0, 2, or 4 pixels).

## Core Idea
The goal is to determine whether pooling choice (architectural bias) still matters once you've already exposed the model to translated versions of the same images (data-driven invariance).

## Findings
- **Augmentation** is the primary driver of translation robustness across all models.
- **MaxPool** provides a small but consistent robustness advantage over **AvgPool**, regardless of the augmentation level.
- The **Null Hypothesis** is supported: architectural choices retain their influence even as you increase training-time augmentation.

## Project Layout
- `src/models.py`: CNN with configurable downsampling blocks.
- `src/data_loader.py`: CIFAR-10 pipeline with padding/cropping.
- `src/engine.py`: Training logic.
- `src/utils.py`: Metric evaluation and deterministic shift transforms.

## How to run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Experiment**:
   ```bash
   python main.py
   ```
   *This will run all 6 experimental conditions and print results.*

---
*Developed by Ibrahim Ahmad*
