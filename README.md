# Eye Movement Trajectory Estimation Using EOG Signals

This repository provides a comprehensive pipeline for estimating 2D eye movement trajectories using four-channel electrooculography (EOG) signals. This project is inspired by research on language embeddings for time series data and the dataset on EOG signals recorded under stationary head conditions.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [References](#references)

## Introduction

Electrooculography (EOG) signals, recorded from four channels, contain valuable information about eye movements. This project aims to estimate the 2D trajectories of eye movements by processing these signals with language-based embeddings and passing them through a deep learning model. This method leverages recent advances in language embedding techniques and convolutional neural networks (CNN) to achieve efficient and accurate trajectory estimations.

## Dataset

The data is based on the study **"Eye Movement Data Recorded Using EOG Under Stationary Head Pose Conditions"** by Barbara et al. [University of Malta], where EOG signals from ten healthy participants were recorded in a controlled setup:
- **Electrodes**: Four electrodes (E1, E2, E3, E4) were placed as shown in the study.
- **Sampling Rate**: 256 Hz, with bandpass filtering between 0-30 Hz and a 50 Hz notch filter to remove line noise.
- **Experimental Protocol**: Each subject performed a series of trials with eye movement tasks and blinks. The dataset includes the horizontal and vertical gaze angles for each target cue displayed on the screen, representing the ground truth for eye movements.

For more details, please refer to the dataset description provided in the repository.

## Methodology

### 1. Signal Embedding Using Language Models

Inspired by the **LETS-C** framework ([Kaur et al., 2024](https://arxiv.org/abs/2407.06533)), which utilizes text embeddings for time series classification, this project processes EOG signals through OpenAI embeddings. Here’s how we handle the EOG signal embeddings:
   - **Embedding Model**: We use the OpenAI Embeddings model to transform EOG signals into high-dimensional embedding vectors, capturing the sequential dependencies of eye movement signals.
   - **Fusion with Original Signals**: To maximize the information fed to the model, we concatenate each original EOG signal with its corresponding embedding. When the embedding length exceeds the original signal length, we apply zero-padding to the original signal.

### 2. Model Architecture

The project adopts a lightweight CNN-based classification head inspired by LETS-C's design principles, ensuring high performance with fewer parameters:
   - **Convolutional Neural Network (CNN)**: The fused embeddings and signals are passed through a 1D CNN that learns spatial hierarchies within each concatenated vector.
   - **Fully Connected Layers**: Flattened outputs from the CNN are passed through fully connected layers (MLP) to map the embeddings to 2D eye trajectory predictions.

### 3. Configuration Options

- `make_embeddings`: Set to `True` to generate embeddings for the EOG signals.
- `make_fusion`: If `True`, original signals are zero-padded to match the embedding length, then fused with embeddings element-wise.
- `batch_size`, `num_workers`: Standard PyTorch Lightning configurations for DataLoader optimization.

## Configuration

The configuration file (`config.yaml`) contains key hyperparameters and options:

```yaml
data_path: "path/to/EOG.mat"
batch_size: 32
num_workers: 4
make_embeddings: True
make_fusion: True
embedding_dim: 3072
