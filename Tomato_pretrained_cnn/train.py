import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
from model import CNN
from data_loader import TomatoLeafDataset
from torch.optim import Adam
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# Define command-line arguments for configuration
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('--ckpt-path', type=str, default='saved_ckpt', help='Checkpoint path')
parser.add_argument('--logs-path', type=str, default='./logs', help='Logs folder path')
parser.add_argument('--cuda', action='store_true', help='If use cuda')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

# Accuracy computation
def accuracy(output, target):
    pred = torch.argmax(output, dim=1)
    correct = pred.eq(target).sum().item()
    return correct

def get_latest_feature_extractor(feature_dir):
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.pth')]
    if not feature_files:
        raise FileNotFoundError(f"No feature extractor files found in {feature_dir}")
    latest_feature = max(feature_files, key=lambda f: os.path.getctime(os.path.join(feature_dir, f)))
    return os.path.join(feature_dir, latest_feature)

# Set the random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility even when using CUDA
    torch.backends.cudnn.benchmark = False  # Disables cuDNN auto-tuner for deterministic behavior

if __name__ == '__main__':
    args = parser.parse_args()

    # Set the random seed
    set_seed(args.seed)

    # Check if CUDA is available
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Create directories if not exist
    os.makedirs(args.ckpt_path, exist_ok=True)
    os.makedirs(args.logs_path, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.logs_path)

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    train_txt = os.path.join(base_path, 'train_labels.txt')
    validate_txt = os.path.join(base_path, 'valid_labels.txt')

    # Load datasets
    train_set = TomatoLeafDataset(base_path, train_txt, image_size=(128, 128), mode='train')
    validate_set = TomatoLeafDataset(base_path, validate_txt, image_size=(128, 128), mode='validate')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    validate_loader = DataLoader(validate_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    
    # Set up model
    model = CNN(num_classes=10, dropout_rate=0.1)
    # Load latest feature extractor weights
    feature_dir = os.path.join(base_path, 'Pretrained_cnn', 'main', 'saved_feature')
    latest_feature_path = get_latest_feature_extractor(feature_dir)
    print(f'Loading feature extractor from: {latest_feature_path}')
    model.features.load_state_dict(torch.load(latest_feature_path, map_location=device))
    model.to(device)  

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_acc = 0
    train_losses, validate_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)

            # Forward pass
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct_train += accuracy(batch_pred, batch_target)
            total_train += batch_target.size(0)
            epoch_train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'[{datetime.now()}] Epoch {epoch+1}/{args.epochs}, Step {batch_idx+1}/{len(train_loader)}, '
                      f'Train Accuracy: {correct_train/total_train:.3f}, Loss: {loss.item():.4f}, Best Val Accuracy: {best_acc:.3f}')
        
        # Log training metrics
        train_losses.append(epoch_train_loss / len(train_loader))
        train_acc = correct_train / total_train
        writer.add_scalar('Train/Loss', train_losses[-1], epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)

        # Validation
        model.eval()
        epoch_val_loss = 0
        correct_validate = 0
        total_validate = 0
        
        with torch.no_grad():
            for batch_data, batch_target in validate_loader:
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                batch_pred = model(batch_data)
                
                loss = criterion(batch_pred, batch_target)
                epoch_val_loss += loss.item()
                correct_validate += accuracy(batch_pred, batch_target)
                total_validate += batch_target.size(0)

        val_acc = correct_validate / total_validate
        validate_losses.append(epoch_val_loss / len(validate_loader))
        writer.add_scalar('Validate/Loss', validate_losses[-1], epoch)
        writer.add_scalar('Validate/Accuracy', val_acc, epoch)
        
        print(f'[{datetime.now()}] Epoch {epoch+1}/{args.epochs}, Validation Accuracy: {val_acc:.3f}')

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, f'best_model_epoch_{epoch+1}.pt'))
            print(f'[{datetime.now()}] Model saved with accuracy: {best_acc:.3f}')


    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss', color='blue')
    plt.plot(range(len(validate_losses)), validate_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss of Tomato Pretrained CNN')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.logs_path, 'loss_curve_tomato_pretrained_cnn.pdf'))
    plt.show()

    # Close TensorBoard writer
    writer.close()