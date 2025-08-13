import torch
import torch.nn as nn
import torch.nn.functional as F

class CSICNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CSICNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        # Placeholder: Will be set after calculating with dummy input
        self._feature_size = None
        self.fc1 = None
        self.fc2 = nn.Linear(128, num_classes)

        self._initialize_fc()

    def _initialize_fc(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 123, 256)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            self._feature_size = x.view(1, -1).shape[1]
            self.fc1 = nn.Linear(self._feature_size, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# For compatibility with main_cnn.py
CNNClassifier = CSICNN
