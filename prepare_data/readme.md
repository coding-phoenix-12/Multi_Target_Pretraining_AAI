# Data Preparation

This directory contains the scripts required to preprocess the datasets used for training and pretraining.

## 1. SPIRE_EMA Corpus (Articulatory Data)

The model requires the SPIRE_EMA corpus for Articulatory-to-Acoustic Inversion (AAI) training.

1.  **Download the dataset** from Hugging Face:
    [https://huggingface.co/datasets/SpireLab/SPIRE_EMA_CORPUS](https://huggingface.co/datasets/SpireLab/SPIRE_EMA_CORPUS)

2.  **Run the processing script**:
    ```bash
    python create_spireema.py
    ```

## 2. Libri360 (Pretraining Data)

For the large-scale pretraining phase, we use the Libri360 dataset.

1.  **Generate Forced Alignments**:
    Use [Kaldi](https://kaldi-asr.org/) to generate forced alignments for your Libri360 audio data.

2.  **Place the Alignment File**:
    Copy your generated alignment file (`FA.txt`) into this directory.
    * *File location:* `data_prep/FA.txt`

3.  **Run the data creation script**:
    ```bash
    python create_data.py
    ```
