import torch
import torch.nn as nn

class Alexnet_ad(nn.Module):
    """Advanced pytorch Alexnet for CIFAR10"""
    def __init__(self, num_classes=1000):
        super(Alexnet_ad, self).__init__()
        self.features = nn.Sequential(
            # (n,3,32,32)
            nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # (n,24,32,32)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (n,24,16,16)
            nn.Conv2d(24, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # (n,96,16,16)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (n.96,8,8)
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # (n,192,8,8)
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # (n,96,8,8)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (n,96,4,4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(96 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),

        )
        self._initializa_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initializa_weights(self):
        """Init weight parameters"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.weight, 0)

