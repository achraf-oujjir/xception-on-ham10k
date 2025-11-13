# Xception on HAM10000 (miracle-skin)

A Jupyter/Colab project that trains an Xception-based image classifier on the HAM10000 skin lesion dataset to distinguish between common pigmented skin lesion types. The notebook in this repository is `miracle-skin.ipynb`.

This README describes the project, expected dataset layout, notebook structure, reproducible training steps, recommended environment, and tips to reproduce or extend the work.

---

## Table of contents

- Project overview
- Files in this repo
- Dataset (HAM10000) — download & expected layout
- Notebook overview (`miracle-skin.ipynb`)
- How the model is built (high level)
- Training procedure (high level)
- Evaluation & outputs
- Reproducing locally or in Colab
- Recommended environment & dependencies
- Tips, caveats and next steps
- Citation & license
- Contact

---

## Project overview

This project demonstrates building a skin lesion classifier using the Xception architecture (transfer learning) on the HAM10000 dataset (seven lesion classes). The primary goal is a clean, reproducible pipeline inside `miracle-skin.ipynb` that performs data loading, preprocessing, augmentation, model definition (Xception backbone + custom head), training with callbacks, and evaluation with common metrics and visualization (confusion matrix, sample predictions).

---

## Files in this repository

- `miracle-skin.ipynb` — primary Jupyter notebook containing data preparation, model creation, training loop, evaluation, plots and notes.
- `README.md` — this file (improved project README).

(If you add scripts, model checkpoints, or data, add them to the repository or store them externally, e.g., Google Drive, and update the notebook paths.)

---

## Dataset — HAM10000

HAM10000 (Human Against Machine with 10000 training images) is a public dataset of dermatoscopic images labelled with 7 diagnostic categories. The dataset is commonly distributed as:

- A CSV file (metadata) containing image identifiers and labels (columns like `image_id`, `dx`).
- A directory of images named like `<image_id>.jpg`.

You can obtain HAM10000 from the original source (ISIC, Harvard/medical group releases, or Kaggle mirrors). Make sure you comply with dataset license and citation requirements.

Expected layout used by the notebook (example):

dataset/
  ├─ images/
  │   ├─ HAM_0000001.jpg
  │   ├─ HAM_0000002.jpg
  │   └─ ...
  └─ metadata.csv  (contains image_id, dx (diagnosis), and optional meta columns)

Notes:
- The notebook expects filenames derived from `image_id` (or you can adapt it to the actual file names).
- If you download the archive from a provider, you might need to extract and rename images or update the CSV accordingly.

---

## Notebook overview (miracle-skin.ipynb)

The notebook follows a typical sequence:

1. Environment setup (imports, GPU check).
2. Load metadata CSV and examine class distribution.
3. Prepare mapping from `image_id` to label (`dx`) and possibly to class indices.
4. Train/validation split — often stratified by label to maintain class balance.
5. Image preprocessing:
   - Resize to model input size (Xception typically expects 299×299, but many projects use 224×224 or 299×299 — check notebook cell for exact size).
   - Normalize pixel values (e.g., scale to [0,1] or use the preprocessing function from `tensorflow.keras.applications.xception`).
   - Data augmentation (flip, rotation, brightness/contrast, random crop). Augmentations are usually applied on-the-fly during training.
6. Model definition:
   - Load Xception as backbone (pretrained on ImageNet) with `include_top=False`.
   - Add a pooling layer (GlobalAveragePooling2D), Dropout for regularization, and final Dense layer with softmax for the number of classes.
7. Compile model:
   - Loss: categorical_crossentropy
   - Optimizer: usually Adam with a small learning rate
   - Metrics: accuracy and optionally AUC
8. Callbacks:
   - ModelCheckpoint (save best weights)
   - ReduceLROnPlateau
   - EarlyStopping (optional)
   - TensorBoard (optional)
9. Training:
   - Fit for a number of epochs with a batch size appropriate for your GPU.
   - Use class weights or balanced sampling if classes are imbalanced.
10. Evaluation:
    - Compute confusion matrix, precision/recall/F1 per class.
    - Plot ROC curves (one-vs-rest) if desired.
    - Show example predictions with images, true & predicted labels.
11. Save model weights and optionally export model for inference.

---

## How the model is built (high level)

- Transfer learning with Xception (pre-trained ImageNet weights).
- Feature extractor: Xception (without the top classification layers).
- Head: GlobalAveragePooling2D -> Dropout -> Dense(num_classes, activation='softmax').
- Fine-tuning: optionally unfreeze some of the top Xception layers after initial training of the head to refine feature weights for dermatoscopic images.

Typical hyperparameters to try:
- Input image size: 299×299 (recommended for Xception) or 224×224 if memory constrained.
- Batch size: 8–64 depending on GPU memory.
- Optimizer: Adam (lr 1e-4 to 1e-5 for fine-tuning).
- Epochs: 20–100 with appropriate callbacks.
- Dropout: 0.2–0.5

---

## Training procedure (recommended, reproducible)

1. Set a fixed random seed for numpy, tensorflow, and python random to improve reproducibility.
2. Prepare a stratified train/validation split (e.g., 80% train, 20% val).
3. Use data augmentation on the training set only.
4. Train in two phases:
   - Phase 1: freeze the Xception base and train the head for a few epochs (e.g., 5–10) with a higher learning rate.
   - Phase 2: unfreeze the last N layers and fine-tune with a lower learning rate (e.g., 1e-5).
5. Use callbacks: ModelCheckpoint (save best by validation loss or metric), ReduceLROnPlateau (monitor val_loss), EarlyStopping.
6. If class imbalance is strong, either:
   - Compute class weights and pass to fit, or
   - Use oversampling / balanced batches in the data generator.

---

## Evaluation & expected outputs

- Training & validation loss/accuracy plots.
- Confusion matrix for the validation set.
- Per-class precision/recall/F1 (classification report).
- Example predicted images (true label vs predicted label and probability).
- Saved best model weights (HDF5 or TensorFlow SavedModel format) in a path you choose.
- Optional: test set performance if you split out a test set.

Note: Because HAM10000 has class imbalance and visual similarity among categories, pay attention to per-class recall and precision instead of raw accuracy.

---

## Reproducing locally or in Google Colab

Colab tips:
- Set Runtime > Change runtime type > Hardware accelerator: GPU.
- Upload `miracle-skin.ipynb` to Colab or open directly from GitHub with the Colab button.
- Mount Google Drive to save models and large files:
  - from google.colab import drive
  - drive.mount('/content/drive')
- Install missing packages at the top of the notebook (if any), e.g.:
  - !pip install -q tensorflow==2.x pandas scikit-learn matplotlib seaborn opencv-python albumentations

Local reproduction:
- Create a virtual environment (venv or conda).
- Install dependencies listed in the "Recommended environment" section below.
- Download and arrange HAM10000 images & CSV to match the notebook paths or adapt the notebook.

Running the notebook headless (script):
- You can convert notebook cells into scripts (e.g., nbconvert) or extract key training and model definition cells into a Python script for programmatic runs.

---

## Recommended environment & dependencies

Minimum suggestions (exact versions used in the notebook may vary):

- Python 3.8–3.11
- TensorFlow 2.10+ (or the version you used in the notebook)
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- pillow or opencv-python
- albumentations (optional, for augmentation)
- tqdm
- jupyter / jupyterlab (if running locally)

Example pip install:
pip install tensorflow pandas scikit-learn matplotlib seaborn pillow albumentations tqdm

If you use GPU with CUDA, install the TensorFlow build compatible with your CUDA/cuDNN.

---

## Tips, caveats and next steps

- Class imbalance: HAM10000 is imbalanced. Use stratified splitting, class weights, or targeted augmentation to improve minority class recognition.
- Data leakage: make sure patient-level splits are used if metadata contains patient identifiers to avoid leakage (images from the same patient appearing in train and val).
- Input size: Xception commonly uses 299×299; smaller sizes reduce memory and speed up training but may reduce accuracy.
- Explainability: use Grad-CAM or similar techniques to visualize what the model focuses on.
- Ensembling: training multiple architectures and ensembling predictions often improves performance.
- External validation: test on external datasets to evaluate generalization.

---

## Citation & license

- HAM10000 dataset and its original publication should be cited when publishing results. See the dataset provider for exact citation text.
- This repository does not include the dataset and does not change the dataset license — please follow HAM10000/ISIC licensing when reusing data.

---

## Contact

Repository owner: achraf-oujjir (GitHub)
If you need clarifications about the notebook internals or want help turning the notebook into a reproducible training script, open an issue or contact the owner through their GitHub profile.

---
