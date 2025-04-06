# Tomato Leaf Disease Detection using Transfer learning
A deep learning-based system for classifying tomato leaf diseases using transfer learning.

## Introduction
This project aims to identify various tomato leaf diseases using deep learning models. It leverages pre-trained CNNs to improve detection accuracy and robustness.

## Project Structure
📁 Tomato-leaf-detection-using-transfer-learning/
├── 📁 Tomato_cnn/                      # CNN trained from scratch
│   ├── 📁 logs/                        # Training logs
│   ├── 📁 saved_ckpt/                 # Model checkpoints
│   ├── 📁 class_confusion_matrices/   # Per-class confusion matrices
│   ├── 📄 confusion_matrix.pdf
│   ├── 📄 roc_curve_tomato_cnn.pdf
│   ├── 📄 model.py
│   ├── 📄 data_loader.py
│   ├── 📄 train.py
│   └── 📄 evaluation.py
│
├── 📁 Pretrained_cnn/                 # Pre-trained CNN on PlantVillage dataset
│   ├── 📁 main/
│   │   ├── 📁 logs/
│   │   ├── 📁 saved_ckpt/
│   │   ├── 📁 saved_feature/
│   │   ├── 📄 confusion_matrix_pretrained_cnn.pdf
│   │   ├── 📄 roc_curve_pretrained_cnn.pdf
│   │   ├── 📄 model.py
│   │   ├── 📄 train.py
│   │   └── 📄 evaluation.py
│   ├── 📄 plantvillage_dataset.zip         # External dataset (not uploaded)
│   ├── 📄 process.py
│   ├── 📄 train_labels.txt
│   ├── 📄 validate_labels.txt
│   └── 📄 test_labels.txt
│
├── 📁 Tomato_pretrained_cnn/         # Transfer learning using pre-trained CNN
│   ├── 📁 logs/
│   ├── 📁 saved_ckpt/
│   ├── 📁 class_confusion_matrices/
│   ├── 📄 confusion_matrix.pdf
│   ├── 📄 roc_curve_tomato_cnn.pdf
│   ├── 📄 model.py
│   ├── 📄 data_loader.py
│   ├── 📄 train.py
│   └── 📄 evaluation.py
│
├── 📄 Tomato_dataset.zip             # Local dataset (not uploaded)
├── 📄 process.py                     # Data processing script
├── 📄 train_labels.txt
├── 📄 validate_labels.txt
├── 📄 test_labels.txt
└── 📄 README.md

## Installation

To install dependencies, run:

```bash
pip install torch
pip install matplotlib
```

## Instruction

To train and evaluate models, use the `train.py` and `evaluation.py` scripts located in each subdirectory (e.g., `Tomato_cnn/`, `Tomato_pretrained_cnn/`).

---

### `train.py` – Model Training

The `train.py` script is used to train a CNN or fine-tune a pre-trained model on the tomato dataset. It supports GPU acceleration with CUDA.

#### Command-line Arguments

| Argument           | Type   | Default             | Description                                          |
|--------------------|--------|---------------------|------------------------------------------------------|
| `--batch-size`     | int    | `64`                | Batch size used during training                      |
| `--lr`             | float  | `0.001`             | Initial learning rate                                |
| `--epochs`         | int    | `150`               | Number of training epochs                            |
| `--ckpt-path`      | str    | `'saved_ckpt'`      | Path to save model checkpoints                       |
| `--feature-path`   | str    | `'saved_feature'`   | Path to save extracted features (if applicable)      |
| `--logs-path`      | str    | `'./logs'`          | Directory to save training logs                      |
| `--cuda`           | flag   | False               | Use GPU (CUDA) for training                          |
| `--seed`           | int    | `42`                | Random seed for reproducibility                      |

#### Example Usage

**Train on CPU:**
```bash
python train.py --batch-size 32 --lr 0.0005 --epochs 100
```

---

### `evaluation.py` – Model Evaluation

The `evaluation.py` script evaluates the trained model using the test dataset. It generates performance metrics such as accuracy, a confusion matrix, and an ROC curve. GPU support is optional.

#### Command-line Arguments

| Argument           | Type   | Default         | Description                                                  |
|--------------------|--------|------------------|--------------------------------------------------------------|
| `--batch-size`     | int    | `64`             | Batch size for evaluation                                    |
| `--cuda`           | flag   | False            | Use CUDA (GPU) if available                                  |
| `--num-classes`    | int    | `2`              | Number of classes in the dataset                             |

> **Note:** You may also need to set model paths and data paths inside the script or add them manually as extra arguments if necessary.

#### Example Usage

**Evaluate on CPU:**
```bash
python evaluation.py --batch-size 64 --num-classes 10
```
