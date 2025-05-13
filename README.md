# Kepler TSFresh + MLP Classification

This repository contains the implementation of a machine learning pipeline for classifying exoplanet candidates using preprocessed Kepler light curves. Statistical time-series features were extracted using the [TSFresh](https://tsfresh.readthedocs.io/) library and used as input to a multilayer perceptron (MLP) and gradient boosted tree (GBT) classifier.

This experiment is part of the Bachelor's thesis **"Machine Learning for Exoplanet Detection: Investigating Feature Engineering Approaches and Stochastic Resonance Effects"** by Alicia Martinelli (2025).

## Folder Structure

```
kepler_array_tsfresh_mlp/
├── raw/                       # Kepler light curves as TFRecords
├── convert_tfrecords.py       # Convert the TFRecords into .npy files and split into train, val and test folders
├── feature_extraction.py      # Feature extraction with TSFresh library to generate the train, val and test datasets
├── gbt.py                     # GBT training and evaluation with Optuna
├── mlp.py                     # MLP training and evaluation with Optuna
├── kepler_gbt.pkl             # Best GBT model
├── kepler_mlp.h3              # Best MLP model
└── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## Preprocessed Kepler dataset
The preprocessed Kepler dataset used in this project is based on the public release from Shallue & Vanderburg (2018) and is available via the AstroNet GitHub repository (Google Drive) [https://drive.google.com/drive/folders/1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE](https://drive.google.com/drive/folders/1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE)

These TFRecord files are already downloaded and placed in the `raw` folder.


## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/AliciaMartinelli/kepler_array_tsfresh_mlp.git
    cd kepler_array_tsfresh_mlp
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    You may need to install `optuna`, `scikit-learn`, `tsfresh`, `matplotlib`, `numpy`, and `tensorflow` (and more).

## Usage

1. Convert TFRecords into .npy files:
```bash
python convert_tfrecords.py
```
This will generate structured .npy arrays and split them into training, validation, and test sets.

2. Extract features from the Kepler `.npy` dataset:
```bash
python feature_extraction.py
```
This step will generate cleaned and scaled TSFresh features for both global and local views.

3. Train and tune the MLP model using Optuna (with evaluation of test set):
```bash
python train_mlp.py
```
The model will be optimized with cross-validation and evaluated on the held-out test set.

4. Train and tune the GBT model using Optuna (with evaluation of test set):
```bash
python train_gbt.py
```
Similar to the MLP step, the GBT model is tuned and evaluated.


## Thesis Context

This repository corresponds to the experiment described in:
- **Section 3.1**: Time-series feature extraction with TSFresh and MLP classification


**Author**: Alicia Martinelli  
**Email**: alicia.martinelli@stud.unibas.ch  
**Year**: 2025
