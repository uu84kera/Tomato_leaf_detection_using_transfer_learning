# Tomato Leaf Disease Detection using Transfer learning
A deep learning-based system for classifying tomato leaf diseases using transfer learning.

## Introduction
This project aims to identify various tomato leaf diseases using deep learning models. It leverages pre-trained CNNs to improve detection accuracy and robustness.

## Project Structure

```
tomato-leaf-detection-using-transfer-learning/
├── Tomato_cnn/                      # CNN trained from scratch
│   ├── logs/
│   ├── saved_ckpt/
│   ├── class_confusion_matrices/
│   ├── confusion_matrix.pdf
│   ├── roc_curve_tomato_cnn.pdf
│   ├── model.py
│   ├── data_loader.py
│   ├── train.py
│   └── evaluation.py
│
├── Pretrained_cnn/                 # Pre-trained CNN on PlantVillage dataset
│   ├── main/
│   │   ├── logs/
│   │   ├── saved_ckpt/
│   │   ├── saved_feature/
│   │   ├── confusion_matrix_pretrained_cnn.pdf
│   │   ├── roc_curve_pretrained_cnn.pdf
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluation.py
│   ├── plantvillage_dataset.zip         # Not uploaded (too large)
│   ├── process.py
│   ├── train_labels.txt
│   ├── validate_labels.txt
│   └── test_labels.txt
│
├── Tomato_pretrained_cnn/         # Transfer learning using pre-trained CNN
│   ├── logs/
│   ├── saved_ckpt/
│   ├── class_confusion_matrices/
│   ├── confusion_matrix.pdf
│   ├── roc_curve_tomato_cnn.pdf
│   ├── model.py
│   ├── data_loader.py
│   ├── train.py
│   └── evaluation.py
│
├── Tomato_dataset.zip             # Not uploaded (too large)
├── process.py
├── train_labels.txt
├── validate_labels.txt
├── test_labels.txt
├── requirements.txt
└── README.md
```
## Dataset

The following datasets are used in this project but not included in the repository due to [GitHub file size limitations](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).  
Please download them manually and unzip into the appropriate folders:

- [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)  
  Used for pretraining the CNN model on general plant leaf disease images.

- [Tomato Leaf Detection Dataset (Kaggle Notebook)](https://www.kaggle.com/code/adinishad/tomato-leaf-detection-by-transfer-learning/notebook)  
  Used for baseline: Tomato_cnn and transfer learning on tomato-specific disease categories: Tomato_pretrained_cnn.

> **Image size:** All input images are resized to `128 × 128` during preprocessing for consistency and computational efficiency.

> **Binary classification in Pretrained CNN:**  
> The `Pretrained_cnn/` model simplifies the problem into **two categories**:
> - `healthy` (class 0, no visible disease) 
> - `unhealthy` (class 1, any disease present)

> **Multiclass classification (Tomato CNN & Transfer Learning):**  
> The `Tomato_dataset.zip` includes **10 distinct classes** of tomato leaf conditions:
> - `Target Spot` (class 0)
> - `Late Blight` (class 1)
> - `Mosaic Virus` (class 2)
> - `Leaf Mold` (class 3)
> - `Bacterial Spot` (class 4)
> - `Early Blight` (class 5)
> - `Healthy` (class 6)
> -  `Yellow Leaf Curl Virus` (class 7)
> -  `Spider Mites (Two-Spotted Spider Mite)` (class 8)
> -  `Septoria Leaf Spot` (class 9)
 
Label mapping is handled by the `process.py` script, which converts folder names or filenames into binary labels.

## Installation

This project requires Python 3.8+ and PyTorch 1.9.0+.

To install all required dependencies, simply run:

```bash
pip install -r requirements.txt
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
Note: All images are resized to 128×128 during training and evaluation.
