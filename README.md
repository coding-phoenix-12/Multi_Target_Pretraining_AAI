# Multi-Target Pretraining for AAI

[![Interspeech 2025](https://img.shields.io/badge/Interspeech-2025-blue.svg)](https://www.isca-archive.org/interspeech_2025/bandekar25_interspeech.pdf)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of the Interspeech 2025 paper: **[Multi-Target Pretraining for Acoustic-to-Articulatory Inversion](https://www.isca-archive.org/interspeech_2025/bandekar25_interspeech.pdf)**.

This repository contains the training and inference pipeline for **Acoustic-to-Articulatory Inversion (AAI)**. The framework utilizes a **FastSpeech-based model** to predict Electromagnetic Articulography (EMA) trajectories from acoustic features.



## 🛠️ Setup

### Prerequisites
Ensure you have a Python environment with **PyTorch** installed. GPU support (CUDA) is highly recommended for training.

### Installation
Install the required dependencies:

```bash
pip install torch numpy pyyaml tqdm
```



### Configuration
All experiment settings are controlled via a YAML configuration file. You do not need to modify the code to change hyperparameters.

### Key Configuration Sections
Before running, update the `data` section in your `config.yaml` to point to your dataset:

```yaml
data:
  rootPath: /path/to/data/root
  featsPath: /path/to/features
```

The config file is organized into the following sections:

* **`data`**: Dataset paths and feature definitions.
* **`type`**: Model architecture settings (e.g., FastSpeech parameters, batch size, epochs).
* **`common`**: Runtime settings (device, dataset selection, input/output dims).
* **`optimizer`**: Learning rate and optimizer specific settings.
* **`earlystopper`**: Early stopping criteria.
* **`logging`**: Experiment logging paths.


## 📂 Data Preparation

For detailed instructions on how to prepare your dataset and features, please refer to the specific documentation:

[**data_prep/README.md**](data_prep/README.md)

### Usage
The entry point for both training and inference is `run.py`. The mode is determined by the `infer` flag in the configuration file.

### 1. Training

To train the model, set the inference flag to `false` in your config:

```yaml
# config.yaml
common:
  infer: false
```
Run the training script:
```bash
python run.py --config path/to/config.yaml
```
###2. Inference
To generate EMA predictions using a trained checkpoint, update the config:

```yaml
# config.yaml
common:
  infer: true
```
Run the same script:

```bash
python run.py --config path/to/config.yaml
```
The model will load the checkpoint saved during training specified by run name in config and generate predictions.

### Citation
If you use this code or findings in your research, please cite our Interspeech 2025 paper:

```bibtex
@inproceedings{bandekar2025enhancing,
  title={Enhancing Acoustic-to-Articulatory Inversion with Multi-Target Pretraining for Low-Resource Settings},
  author={Bandekar, Jesuraj and Ghosh, Prasanta Kumar},
  booktitle={Proc. Interspeech 2025},
  pages={5588--5592},
  year={2025}
}
```
