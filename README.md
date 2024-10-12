# Multi-label-Classification-of-Chest-X-rays-for-Disease-Detection



This repository contains a Jupyter Notebook that performs multi-label classification of chest X-rays using a deep learning convolutional neural network (CNN). The dataset consists of X-ray images labeled with various thoracic diseases. The model is built using TensorFlow and Keras and includes data preprocessing, data augmentation, and multiple convolutional layers for feature extraction.

## Project Overview

The notebook implements a multi-label classification task using chest X-ray images to detect several diseases such as:

- Hernia
- Pneumonia
- Fibrosis
- Edema
- Emphysema
- Cardiomegaly
- Pleural Thickening
- Consolidation
- Pneumothorax
- Mass
- Nodule
- Atelectasis
- Effusion
- Infiltration
- No Finding

The primary objective is to classify an image with one or more of these labels.

### Model Architecture

The architecture includes:
- **Multiple Convolutional Layers** for feature extraction
- **Batch Normalization**, **Dropout**, and **Dense Layers** for fully connected layers
- **Sigmoid Activation** for multi-label classification

The model is trained using **Adam Optimizer** and **Binary Crossentropy** loss, and leverages callbacks such as **Early Stopping**, **Model Checkpoint**, and **ReduceLROnPlateau** to monitor validation loss and adjust learning rate dynamically.

## Dataset

The dataset used in this project is the **NIH Chest X-ray Dataset** (Sample) available on Kaggle.

You can download the dataset from this link: [NIH Chest X-rays Dataset (Sample)](https://www.kaggle.com/datasets/nih-chest-xrays/sample)

This dataset contains chest X-ray images and their associated disease labels, with each image potentially having multiple labels.

### Example Dataset Structure

The dataset is stored in CSV format with the following structure:
- `Image Index`: Image file names
- `Patient Gender`: Gender of the patient
- `Finding Labels`: List of diseases associated with the image (multi-label format)

The dataset is available as a CSV file and corresponding image folder:
- CSV file: `sample_labels.csv`
- Image folder: `sample/images/`

## Project Structure

- **`chest_xray_classification.ipynb`**: Jupyter Notebook that contains the full pipeline, including data preprocessing, model training, and evaluation.
- **`README.md`**: Instructions and project description.

## Data Augmentation

To improve model generalization, data augmentation is applied:
- **Rescaling**
- **Samplewise Standardization and Centering**
- **Random Rotations**
### Callbacks:
- **EarlyStopping**: Monitors `val_loss` and stops training if no improvement after 5 epochs.
- **ModelCheckpoint**: Saves the best model during training.
- **ReduceLROnPlateau**: Reduces the learning rate if the validation loss plateaus.

## Model Training

The model is trained using a convolutional neural network (CNN) architecture. It includes:
- **6 Convolutional Layers** with MaxPooling.
- **Fully Connected Layers** with Batch Normalization and Dropout to reduce overfitting.
- The model uses **Binary Crossentropy** as the loss function and **Sigmoid Activation** in the final layer for multi-label classification.



## Evaluation

The model is evaluated on both the training and validation sets using accuracy metrics and loss plots to monitor the performance.

---

This version includes the link to the **NIH Chest X-rays Dataset (Sample)** from Kaggle. You can modify the repository name and structure based on your actual setup.
