from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

"""
This is a file for loading the tomato leaf dataset.
"""

class TomatoLeafDataset(Dataset):
    def __init__(self, data_path, txt_path, image_size, mode='train'):
        self.data_path = data_path
        self.txt_path = txt_path
        self.mode = mode
        # Read image paths, labels, etc. from the text file
        self.im_lst, self.labels = [], []
        with open(self.txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')  # Split each line by commas
                if len(parts) == 2:
                    img_path, label = parts
                    self.im_lst.append(img_path.strip())  # Remove leading/trailing spaces
                    label = int(label.strip())  # Ensure label is an integer
                    self.labels.append(label)

        assert len(self.im_lst) == len(self.labels), "Mismatch between image paths and labels"
        
        self.image_size = image_size
        # Define image transformation operations
        if self.mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size), 
                transforms.RandomHorizontalFlip(),  # Random flip for augmentation (train only)
                transforms.RandomRotation(90),     # Random rotation for augmentation (train only)
                transforms.ToTensor(), 
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size), 
                transforms.ToTensor(), 
            ])


    def __getitem__(self, idx):
        # Use os.path.join to construct image paths and avoid extra slashes
        image_path = os.path.join(self.data_path, self.im_lst[idx].lstrip('/'))  # Remove leading slashes
        im = Image.open(image_path)
        label = self.labels[idx]
        # Apply transformation to the image
        im = self.transform(im)
        return im, label

    def __len__(self):
        return len(self.im_lst)


if __name__ == '__main__':

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #print(base_path)

    # Define full paths for the txt files inside the dataset directory
    train_txt = os.path.join(base_path, 'train_labels.txt')
    validate_txt = os.path.join(base_path, 'valid_labels.txt')
    test_txt = os.path.join(base_path, 'test_labels.txt')

    # Load the training set
    dataset = TomatoLeafDataset(base_path, train_txt, image_size=(128, 128), mode='train')
    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)  # Ensure batches are full
    for i, (im, label) in enumerate(train_loader):
        print(f"Batch {i+1}: Image Size: {im.shape}, Label Size: {label.shape}")
    
    # Load the validation set (no shuffling, no drop_last)
    dataset = TomatoLeafDataset(base_path, validate_txt, image_size=(128, 128), mode='validate')
    validate_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, num_workers=0)
    for i, (im, label) in enumerate(validate_loader):
        print(f"Validation Batch {i+1}: Image Size: {im.shape}, Label Size: {label.shape}")
    
    # Load the test set (no shuffling, no drop_last)
    dataset = TomatoLeafDataset(base_path, test_txt, image_size=(128, 128), mode='test')
    test_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, num_workers=0)
    for i, (im, label) in enumerate(test_loader):
        print(f"Test Batch {i+1}: Image Size: {im.shape}, Label Size: {label.shape}")

