# Skin Lesion Classification with ISIC 2018

## Overview

This Jupyter Notebook implements a skin lesion classification pipeline using the **ISIC 2018** dataset. It leverages transfer learning with two state-of-the-art deep convolutional neural network architectures—**ResNet50** and **EfficientNetB0**—to distinguish between different melanoma classes.

The notebook is organized into modular sections, from data loading and preprocessing to model training, fine-tuning, and evaluation. It produces insightful visualizations including training/validation accuracy and loss curves, confusion matrices, and classification reports.

## Notebook Sections

* **Section A: Setup and Imports**
  Installs and imports required libraries (TensorFlow, Pandas, OpenCV, scikit-learn, Matplotlib).

* **Section B: Load Metadata and Image Paths**
  Reads the ISIC 2018 metadata CSV, constructs full file paths, and filters out incomplete entries.

* **Section C: Train/Validation/Test Split**
  Creates stratified splits to maintain class balance across training, validation, and test sets.

* **Section D: Preprocess and Resize Images**
  Resizes all lesion images to 224×224 pixels and saves them to a structured directory for fast loading.

* **Section E: Prepare Final DataFrames by Split**
  Generates separate DataFrames (`train_df`, `val_df`, `test_df`) including labels and image file paths.

* **Section F: Image Generators (ResNet Compatible)**
  Defines `ImageDataGenerator` pipelines with real-time augmentation (rotation, flipping, brightness shifts).

* **Section G: Model Architectures (ResNet50 & EfficientNetB0)**

  * **ResNet50**: Uses `tf.keras.applications.ResNet50` with custom top layers.
  * **EfficientNetB0**: Uses `tf.keras.applications.EfficientNetB0` with custom classifier head.

* **Section H: Train ResNet50**

  1. **Frozen Base**: Trains only the added classification head.
  2. **Fine-Tuning**: Unfreezes the last 2 convolutional blocks and continues training.
     Monitors performance via early stopping and learning-rate scheduling.

* **Section I: EfficientNet-Compatible Data Generators**
  Adjusts preprocessing functions compatible with EfficientNet’s requirements.

* **Section J: Train EfficientNetB0**
  Two-phase training identical to ResNet50: head-only, then fine-tuning.

* **Section K: Evaluation and Comparison**

  * Plots **training vs. validation** accuracy and loss curves.
  * Computes **confusion matrices** for the test set.
  * Generates **classification reports** (precision, recall, F1-score).

## Example Outputs

* **Training Curves**: Both models converge within 20–30 epochs, reaching training accuracy above 90% and validation accuracy around 85–90%.
* **Confusion Matrix**: Detailed breakdown of true vs. predicted classes highlights areas of confusion between melanoma subtypes.
* **Classification Report**: Per-class precision and recall metrics; overall F1-score demonstrates strong performance.

## Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/skin-lesion-classification.git
   cd skin-lesion-classification
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**

   ```bash
   jupyter notebook notebooke3168261fb.ipynb
   ```

4. **View Results**

   * Plots are generated inline.
   * Saved model weights (`.h5`) in the `models/` directory.

## Next Steps (Work in progress)

The next phase of this project is to **simulate an embedded systems deployment**, enabling the trained models to run on resource-constrained hardware. This involves:

* Converting models to **TensorFlow Lite** or **ONNX** format.
* Building a **hardware-in-the-loop** simulation targeting microcontrollers or edge devices.
* Evaluating performance (latency, memory footprint, power consumption) in an emulated real-device scenario.

## File Structure

```
├── notebooke3168261fb.ipynb      # Main analysis notebook
├── models/                       # Saved model weights (ResNet50, EfficientNetB0)
├── requirements.txt             # Python dependencies
└── README.md                     # This document
```
