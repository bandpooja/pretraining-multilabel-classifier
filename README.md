**Unsupervised e-Learning with Pretrained CNNs and VLMs**

## Overview
This repository implements an unsupervised e-learning approach by pretraining Convolutional Neural Networks (CNNs) and Vision-Language Models (VLMs). The framework includes data loaders and trainers for the following self-supervised learning models:

- **MoCo (Momentum Contrast)**
- **BYOL (Bootstrap Your Own Latent)**
- **SimSiam (Simple Siamese Networks)**

These models enable efficient feature learning from unlabeled data, improving downstream tasks such as classification, clustering, and retrieval.

## Features
- Modular implementation of **MoCo, BYOL, and SimSiam**.
- Efficient **data loaders** for large-scale image datasets.
- Customizable **trainer** scripts for flexible training configurations.
- Support for **pretrained CNN backbones** (e.g., ResNet, EfficientNet).
- Integration with **Vision-Language Models (VLMs)** for multi-modal learning.
- **Distributed training** support using PyTorch.
