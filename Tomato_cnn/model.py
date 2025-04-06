import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""
This is the convolutional neural network for Tomato leaf disease detection.
"""

class CNN(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(CNN, self).__init__()
        
        # Convolution Layer1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)  # 3*3 kernel
        # Convolution Layer2
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)  # 3*3 kernel
        # Convolution Layer3
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0)  # 3*3 kernel

        # Max Pooling 
        self.pool = nn.MaxPool2d(2, 2)  # 2*2 kernel

        # Fully connected layer
        self.fc1 = nn.Linear(1568, 128)
        # output
        self.output = nn.Linear(128, num_classes)
        
        # Dropout layer (you can modify the dropout rate here)
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights with Glorot uniform
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))  # First conv + ReLU
        #print("layer1 shape ", x.shape)
        x = self.pool(x)
        #print("layer2 shape ", x.shape)
        x = F.relu(self.conv2(x))
        #print("layer3 shape ", x.shape)
        x = self.pool(x)
        #print("layer4 shape ", x.shape)
        x = F.relu(self.conv3(x))
        #print("layer3 shape ", x.shape)
        x = self.pool(x)
        x = x.flatten(1)
        #print("layer4 shape ", x.shape)
        x = F.relu(self.fc1(x))
        #print("layer5 shape ", x.shape)
        x = self.output(x)
        #print("layer6 shape ", x.shape)
        return x
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # This is Glorot (Xavier) initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Bias is initialized to zero