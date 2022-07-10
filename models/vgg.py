import torch
from torch import nn
from .utils import get_state_dict


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(arch, cfg, ckpt_path, **kwargs):
    if ckpt_path:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg]), **kwargs)
    if ckpt_path:
        state_dict = get_state_dict(ckpt_path)
        model.load_state_dict(state_dict)
    return model


def vgg11(ckpt_path=None, num_classes=1000):
    return _vgg("vgg11", "A", ckpt_path, num_classes=num_classes)


def vgg13(ckpt_path=None, num_classes=1000):
    return _vgg("vgg13", "B", ckpt_path, num_classes=num_classes)


def vgg16(ckpt_path=None, num_classes=1000):
    return _vgg("vgg16", "D", ckpt_path, num_classes=num_classes)


def vgg19(ckpt_path=None, num_classes=1000):
    return _vgg("vgg19", "E", ckpt_path, num_classes=num_classes)


###################################Tiny########################################


class VGGTiny(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGGTiny, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _vgg_tiny(arch, ckpt_path, **kwargs):
    if ckpt_path:
        kwargs["init_weights"] = False
    cfg = [16, "M", 32, 32, "M", 64, 64, 64, "M", 128]
    model = VGGTiny(make_layers(cfg), **kwargs)
    if ckpt_path:
        state_dict = get_state_dict(ckpt_path)
        model.load_state_dict(state_dict)
    return model


def vgg_tiny(ckpt_path=None, num_classes=10):
    return _vgg_tiny("vgg_tiny", ckpt_path, num_classes=num_classes)
