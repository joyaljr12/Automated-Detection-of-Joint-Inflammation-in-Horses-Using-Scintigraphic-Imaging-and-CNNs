import torch
import torch.nn as nn


class FTUCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1 ), # in : (1, 128, 128)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # out: (16, 64, 64) 

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1 ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # out: (32, 32, 32) 

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1 ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # out: (64, 16, 16) 

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)

        )

    def forward(self, X):
        return self.network(X)
