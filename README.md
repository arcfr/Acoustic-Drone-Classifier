# Acoustic-Drone-Classifier

**Author:** Archit Sahay

## Overview

Acoustic-Drone-Classifier is a deep learning framework for multi-class drone classification using acoustic signatures extracted from environmental audio recordings.

The project investigates drone recognition through time-frequency audio representations and benchmarks a custom 2D Convolutional Neural Network against a Transfer Learning pipeline based on a pre-trained ResNet18 architecture.

The framework performs four-class classification across the following categories:

- Light Drone
- Medium Drone
- Heavy Drone
- No Drone

The No-Drone category consists of environmental background recordings including traffic noise, wind, rainfall, bird vocalizations, and other ambient sounds.

---

## Project Objective

The objective of this project is to investigate whether environmental audio recordings can be used to reliably detect and classify drone activity using deep learning techniques.

The framework processes raw audio streams, extracts discriminative time-frequency representations, and performs multi-class classification based on acoustic characteristics observed across different drone categories.

---

## Dataset

The dataset was manually curated by Archit Sahay from publicly available drone recordings and environmental audio sources.

Audio segments were manually inspected, verified, labeled, and categorized before feature extraction.

To improve acoustic diversity, recordings were collected from multiple independent sources spanning different recording environments, drone platforms, microphone characteristics, and background conditions.

### Dataset Composition

| Class | Samples |
|---------|---------:|
| Light Drone | 399 |
| Medium Drone | 370 |
| Heavy Drone | 426 |
| No Drone | 449 |
| **Total** | **1644** |

---

## Performance Benchmark

Evaluation was conducted using:

- 25% Test Split
- 10% Validation Split (derived from the training set)
- Randomized dataset shuffling
- Early stopping based on validation loss monitoring
- Best-model checkpoint restoration

### Performance Comparison

Evaluation was performed on a held-out test set consisting of 411 samples.

| Metric | Custom 2D-CNN (MFCC + Deltas) | ResNet18 (Log-Mel Spectrograms) |
|----------|----------:|----------:|
| Overall Accuracy | 88.0% | **95.0%** |
| Macro F1-Score | 0.88 | **0.95** |
| Weighted F1-Score | 0.88 | **0.95** |
| Light Precision | 0.92 | **0.94** |
| Light Recall | 0.92 | **0.94** |
| Light F1-Score | 0.92 | **0.94** |
| Medium Precision | 0.92 | **0.94** |
| Medium Recall | 0.76 | **0.96** |
| Medium F1-Score | 0.84 | **0.95** |
| Heavy Precision | 0.80 | **0.94** |
| Heavy Recall | 0.93 | **0.96** |
| Heavy F1-Score | 0.86 | **0.95** |
| No-Drone Precision | 0.92 | **0.97** |
| No-Drone Recall | 0.90 | **0.93** |
| No-Drone F1-Score | 0.91 | **0.95** |

---

### Classification Report Summary

#### Custom 2D-CNN (MFCC + Delta Features)

| Class | Precision | Recall | F1-Score | Support |
|---------|---------:|---------:|---------:|---------:|
| Light | 0.92 | 0.92 | 0.92 | 100 |
| Medium | 0.92 | 0.76 | 0.84 | 93 |
| Heavy | 0.80 | 0.93 | 0.86 | 106 |
| No-Drone | 0.92 | 0.90 | 0.91 | 112 |

| Metric | Value |
|----------|----------:|
| Accuracy | 0.88 |
| Macro Avg F1 | 0.88 |
| Weighted Avg F1 | 0.88 |

#### ResNet18 (Log-Mel Spectrograms)

| Class | Precision | Recall | F1-Score | Support |
|---------|---------:|---------:|---------:|---------:|
| Light | 0.94 | 0.94 | 0.94 | 100 |
| Medium | 0.94 | 0.96 | 0.95 | 93 |
| Heavy | 0.94 | 0.96 | 0.95 | 106 |
| No-Drone | 0.97 | 0.93 | 0.95 | 112 |

| Metric | Value |
|----------|----------:|
| Accuracy | 0.95 |
| Macro Avg F1 | 0.95 |
| Weighted Avg F1 | 0.95 |

---

## Experimental Findings

- The ResNet18 Transfer Learning pipeline consistently outperformed the custom 2D-CNN across all evaluation metrics.
- Log-Mel Spectrogram representations provided stronger class separation than MFCC-based features.
- Medium Drone classification represented the most challenging category for the custom CNN, suggesting acoustic overlap between medium-class drones and adjacent categories.
- Heavy Drone and No-Drone classes demonstrated strong separability across both architectures.
- Early stopping improved generalization performance by preventing late-stage overfitting and restoring the best validation checkpoint.

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
22050 Hz
```

to eliminate collection-device inconsistencies and maintain feature uniformity.

### 3. Feature Extraction

Two independent feature generation pipelines are implemented.

#### MFCC Pipeline

Generated channels:

- MFCC
- Delta MFCC
- Delta-Delta MFCC

#### Log-Mel Spectrogram Pipeline

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

### Step 1: Extract Audio

```bash
python scripts/extract_audio.py
```

### Step 2: Preprocess Audio

```bash
python scripts/preprocess_audio.py
```

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

- Stratified dataset splitting
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