# Acoustic-Drone-Classifier

**Author:** Archit Sahay

## Overview

Acoustic-Drone-Classifier helps with classification over drone audio recordings developed for multi-class drone classification using image classification.

The project evaluates two independent feature-learning pipelines:

1. A custom 2D Convolutional Neural Network trained on MFCC, Delta MFCC, and Delta-Delta MFCC representations.
2. A Transfer Learning approach utilizing a pre-trained ResNet18 backbone operating on Log-Mel Spectrograms.

The framework performs four-class classification across the following categories:

- Light Drone
- Medium Drone
- Heavy Drone
- No Drone

---

## Project Objective

The objective of this project is to investigate whether environmental audio recordings can be used to reliably detect and classify drone activity using deep learning techniques.

The framework processes raw audio streams, extracts discriminative time-frequency representations, and performs multi-class classification based on acoustic characteristics observed across different drone categories.

---

## Performance Benchmark

Evaluation was conducted using:

- 25% Test Split
- 10% Validation Split (derived from the training set)
- Randomized dataset shuffling
- Early stopping based on validation loss monitoring

## Performance Comparison

Evaluation was performed on a held-out test set consisting of 411 samples.

| Metric             | Custom 2D-CNN (MFCC + Deltas) | ResNet18 (Log-Mel Spectrograms) |
| ------------------ | ----------------------------: | ------------------------------: |
| Overall Accuracy   |                     **90.0%** |                       **92.0%** |
| Macro F1-Score     |                      **0.89** |                        **0.92** |
| Weighted F1-Score  |                      **0.90** |                        **0.92** |
| Light Precision    |                      **0.93** |                        **0.99** |
| Light Recall       |                      **0.85** |                        **0.88** |
| Light F1-Score     |                      **0.89** |                        **0.93** |
| Medium Precision   |                      **0.82** |                        **0.89** |
| Medium Recall      |                      **0.85** |                        **0.90** |
| Medium F1-Score    |                      **0.84** |                        **0.90** |
| Heavy Precision    |                      **0.88** |                        **0.94** |
| Heavy Recall       |                      **0.94** |                        **0.93** |
| Heavy F1-Score     |                      **0.91** |                        **0.94** |
| No-Drone Precision |                      **0.94** |                        **0.89** |
| No-Drone Recall    |                      **0.93** |                        **0.97** |
| No-Drone F1-Score  |                      **0.93** |                        **0.93** |

### Classification Report Summary

#### Custom 2D-CNN (MFCC + Delta Features)

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------: | -----: | -------: | ------: |
| Light    |      0.93 |   0.85 |     0.89 |     100 |
| Medium   |      0.82 |   0.85 |     0.84 |      93 |
| Heavy    |      0.88 |   0.94 |     0.91 |     106 |
| No-Drone |      0.94 |   0.93 |     0.93 |     112 |

#### ResNet18 (Log-Mel Spectrograms)

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------: | -----: | -------: | ------: |
| Light    |      0.99 |   0.88 |     0.93 |     100 |
| Medium   |      0.89 |   0.90 |     0.90 |      93 |
| Heavy    |      0.94 |   0.93 |     0.94 |     106 |
| No-Drone |      0.89 |   0.97 |     0.93 |     112 |


---

## Experimental Findings

### Early Stopping and Generalization

Both architectures exhibited increasing validation instability when trained for a fixed number of epochs.

To address this, an early stopping mechanism was incorporated using validation loss monitoring and checkpoint restoration of the best-performing model state.

This reduced overfitting and improved holdout-set performance across both architectures.

### Model Comparison

#### Custom 2D-CNN

Input Features:

- MFCC
- Delta MFCC
- Delta-Delta MFCC

Characteristics:

- Lower computational overhead
- Faster training and inference
- Suitable for resource-constrained environments

#### ResNet18 Transfer Learning

Input Features:

- Log-Mel Spectrograms

Characteristics:

- Higher classification performance
- Better feature abstraction
- Increased computational requirements

---

## Dataset

The dataset was manually curated by Archit Sahay from publicly available drone recordings and environmental audio sources.

Audio segments were manually inspected, verified, labeled, and categorized before feature extraction.

Current classes include:

- Light Drone
- Medium Drone
- Heavy Drone
- No Drone

To improve acoustic diversity, recordings were collected from multiple independent sources spanning different recording environments, drone platforms, microphone characteristics, and background conditions.

---

## Data Processing Pipeline

### 1. Audio Segmentation

Long-duration recordings are segmented into fixed-length 2-second audio clips to ensure uniform model inputs.

### 2. Audio Preprocessing

Each audio sample undergoes a standardized preprocessing pipeline:

- Silence trimming
- Peak amplitude normalization
- Sampling rate standardization

All recordings are resampled to:

```text
22,050 Hz
```

to eliminate collection-device inconsistencies and maintain feature uniformity.

### 3. Feature Extraction

Two independent feature generation pipelines are implemented.

#### MFCC Pipeline

Generated using:

```bash
python scripts/extract_mfcc.py
```

Generated channels:

- MFCC
- Delta MFCC
- Delta-Delta MFCC

#### Log-Mel Spectrogram Pipeline

Generated using:

```bash
python scripts/extract_spectrograms.py
```

Generated representation:

- Log-Mel Spectrogram Images

---

## Project Structure

```text
Acoustic-Drone-Classifier/
├── scripts/
│   ├── extract_audio.py
│   ├── preprocess_audio.py
│   ├── extract_mfcc.py
│   └── extract_spectrograms.py
│
├── training_scripts/
│   ├── train_cnn.py
│   └── train_resnet.py
│
├── video_metadata.json
├── .gitignore
└── README.md
```

---

## Installation

Install the required dependencies:

```bash
pip install torch torchvision numpy librosa scikit-learn
```

---
## Audio Generation 

Step 1. run scripts/extract_audio.py 
Step 2. Run scripts/preprocess_audio.py

---
## Feature Generation

### MFCC Feature Extraction

```bash
python scripts/extract_mfcc.py
```

### Log-Mel Spectrogram Generation

```bash
python scripts/extract_spectrograms.py
```

---

## Training and Evaluation

### Custom 2D-CNN

```bash
python training_scripts/train_cnn.py
```

### ResNet18 Transfer Learning

```bash
python training_scripts/train_resnet.py
```

Both training pipelines include:

- Dataset splitting
- Validation monitoring
- Early stopping
- Best-model checkpoint restoration
- Classification report generation
- Confusion matrix evaluation

---

## Author

**Archit Sahay**

B.Tech Information Technology

Summer Internship Project

Acoustic Drone Detection and Classification using Deep Learning