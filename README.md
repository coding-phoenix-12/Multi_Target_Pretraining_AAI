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



##  Data Preparation

For detailed instructions on how to prepare your dataset and features, please refer to the specific documentation:

[**prepare_data/README.md**](prepare_data/README.md)

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

The framework automatically selects the appropriate model architecture based on the `predict` targets defined in your config and the `run_name`.

| Scenario | Config Settings | Model Class Used |
| :--- | :--- | :--- |
| **Baseline AAI** | `common.predict` includes `"ema"` <br> AND `logging.run_name` includes `"baseline"` | `baseline.FastSpeech` |
| **Fine-tuning** | `common.predict` includes `"ema"` <br> AND `logging.run_name` does **not** include `"baseline"` | `finetune.FastSpeechft` |
| **Pretraining** | `common.predict` includes any of: <br> `["place", "manner", "height", "backness", "phones", "weights"]` | `pretrain.FastSpeech` |


### Example Configs

**1. To run the Baseline AAI model:**
```yaml
common:
  predict: ["ema"]
logging:
  run_name: "my_baseline_experiment"
```

**2. To run Pretraining (e.g., on Place of Articulation):**

```yaml
common:
  predict: ["place"]
```

### Advanced Data & Model Control

These parameters in `config.yaml` allow you to simulate low-resource settings and control fine-tuning:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`exclude_utts`** | `list` | **Low-Resource Simulation:** Controls which utterances or what percentage of data to use. This is used to simulate low-resource settings by training on only a fraction of the dataset. |
| **`exclude_odd`** | `bool` | **Data Split Control:** If set to `true`, it excludes odd-numbered utterances from the training set (often used for specific train/test splits). |
| **`load_models`** | `list` | **Fine-tuning Checkpoint:** Specifies the name(s) of the pretrained model checkpoint(s) to load when starting a fine-tuning run. |

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
