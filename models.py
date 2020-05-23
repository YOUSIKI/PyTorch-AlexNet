import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=(4 if kernel_size == 11 else kernel_size // 2)
        )


class LocalResponseNorm(nn.LocalResponseNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(size=5, alpha=1e-4, beta=0.75, k=2.0)


class BatchNorm(nn.BatchNorm2d):

    def __init__(self, channels, *args, **kwargs):
        super().__init__(num_features=channels)


class Identity(nn.Identity):

    def __init__(self, *args, **kwargs):
        super().__init__()


class AlexNet(nn.Module):

    def __init__(
        self,
        num_classes=1000,
        conv=Conv2d,
        activation=nn.ReLU,
        normalization=LocalResponseNorm,
        pooling=nn.MaxPool2d
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conv = conv
        self.pooling = pooling
        self.activation = activation
        self.normalization = normalization
        self.layers = nn.ModuleList()
        self.layers.append(self.conv(3, 96, 11, 4))
        self.layers.append(self.activation())
        self.layers.append(self.normalization(96))
        self.layers.append(self.pooling(3, 2))
        self.layers.append(self.conv(96, 256, 5, groups=2))
        self.layers.append(self.activation())
        self.layers.append(self.normalization(256))
        self.layers.append(self.pooling(3, 2))
        self.layers.append(self.conv(256, 384, 3))
        self.layers.append(self.activation())
        self.layers.append(self.conv(384, 384, 3, groups=2))
        self.layers.append(self.activation())
        self.layers.append(self.conv(384, 256, 3, groups=2))
        self.layers.append(self.activation())
        self.layers.append(self.pooling(3, 2))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Dropout())
        self.layers.append(nn.Linear(256 * 6 * 6, 4096))
        self.layers.append(self.activation())
        self.layers.append(nn.Dropout())
        self.layers.append(nn.Linear(4096, 4096))
        self.layers.append(self.activation())
        self.layers.append(nn.Dropout())
        self.layers.append(nn.Linear(4096, self.num_classes))

    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
