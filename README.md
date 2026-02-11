# Multi_Target_Pretraining_AAI
Implementation for the interspeech paper: 

https://www.isca-archive.org/interspeech_2025/bandekar25_interspeech.pdf

AAI Training Framework

This repository contains the training and inference pipeline for Articulatory-to-Acoustic Inversion (AAI) using a FastSpeech-based model.
The model predicts EMA trajectories from acoustic features.

Setup

Install dependencies:

pip install torch numpy pyyaml tqdm


Ensure CUDA is available if using GPU.

Configuration

All settings are controlled via a YAML config file.

Before running, update:

data:
  rootPath: <root_path>
  featsPath: <path_to_features>


Key sections in the config:

data – dataset and feature paths

type – model settings (batch size, epochs, architecture)

common – runtime settings (device, dataset, inputs, outputs)

optimizer – optimizer settings

earlystopper – early stopping

logging – experiment logging

Training

Set in config:

common:
  infer: false


Then run:

python run.py --config path/to/config.yaml

Inference

To run inference, simply set:

common:
  infer: true


Then run the same script:

python run.py --config path/to/config.yaml


The model will load the specified checkpoint and generate EMA predictions.
