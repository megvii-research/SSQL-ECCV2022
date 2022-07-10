import torch
import torch.nn as nn

__all__ = ["vgg_tiny_bnn"]


class VGG_Cifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_Cifar10, self).__init__()
        self.infl_ratio = 3
        self.features = nn.Sequential(
            nn.Conv2d(3, 128 * self.infl_ratio, kernel_size=3, padding=1),
            nn.BatchNorm2d(128 * self.infl_ratio),
            nn.Hardtanh(inplace=True),
            nn.Conv2d(
                128 * self.infl_ratio, 128 * self.infl_ratio, kernel_size=3, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128 * self.infl_ratio),
            nn.Hardtanh(inplace=True),
            nn.Conv2d(
                128 * self.infl_ratio, 256 * self.infl_ratio, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(256 * self.infl_ratio),
            nn.Hardtanh(inplace=True),
            nn.Conv2d(
                256 * self.infl_ratio, 256 * self.infl_ratio, kernel_size=3, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256 * self.infl_ratio),
            nn.Hardtanh(inplace=True),
            nn.Conv2d(
                256 * self.infl_ratio, 512 * self.infl_ratio, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(512 * self.infl_ratio),
            nn.Hardtanh(inplace=True),
            nn.Conv2d(512 * self.infl_ratio, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            nn.Linear(1024, num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


def vgg_tiny_bnn(ckpt_path=None, num_classes=10):
    model = VGG_Cifar10(num_classes=num_classes)
    if ckpt_path:
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict)
    return model
