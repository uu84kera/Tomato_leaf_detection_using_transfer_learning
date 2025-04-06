import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
from model import CNN
from data_loader import TomatoLeafDataset

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
parser.add_argument('--num-classes', type=int, default=10, help='Number of classes in the dataset')
args = parser.parse_args()

def get_latest_checkpoint(ckpt_dir):
    checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getctime(os.path.join(ckpt_dir, f)))
    return os.path.join(ckpt_dir, latest_checkpoint)

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_probs.append(probabilities.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    return np.array(all_preds), np.array(all_labels), all_probs

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix_per_class(y_true, y_pred, class_idx, class_name, save_dir):
    # Binary classification: class vs. all others
    y_true_binary = (y_true == class_idx).astype(int)
    y_pred_binary = (y_pred == class_idx).astype(int)
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not ' + class_name, class_name],
                yticklabels=['Not ' + class_name, class_name])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for Class {class_name}')
    
    save_path = os.path.join(save_dir, f'confusion_matrix_class_{class_name}.pdf')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, num_classes, save_path):
    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_txt = os.path.join(base_path, 'test_labels.txt')
    dataset = TomatoLeafDataset(base_path, test_txt, image_size=(128, 128), mode='test')
    test_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Get the latest checkpoint file
    path = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(path, 'saved_ckpt')
    latest_ckpt = get_latest_checkpoint(ckpt_path)
    
    # Load the model
    model = CNN(num_classes=args.num_classes, dropout_rate=0.5).to(device)
    model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    
    # Evaluate model
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device, args.num_classes)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    
    # Save overall confusion matrix
    cm_save_path = os.path.join(path, 'confusion_matrix.pdf')
    plot_confusion_matrix(y_true, y_pred, classes=[str(i) for i in range(args.num_classes)], save_path=cm_save_path)

    # Save individual class confusion matrices
    save_dir = os.path.join(path, 'class_confusion_matrices')
    os.makedirs(save_dir, exist_ok=True)

    for i in range(args.num_classes):
        plot_confusion_matrix_per_class(y_true, y_pred, i, class_name=str(i), save_dir=save_dir)

    # Optionally save ROC curve
    roc_save_path = os.path.join(path, 'roc_curve.pdf')
    plot_roc_curve(y_true, y_probs, args.num_classes, save_path=roc_save_path)
    