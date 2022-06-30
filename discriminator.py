from torch import nn
from collections import OrderedDict


class Discriminator(nn.Module):
    def __init__(self, input_channels, h=224, w=224):
        super(Discriminator, self).__init__()
        self.h = h
        self.w = w
        self.discriminator = get_discriminator(input_channels=input_channels, final_sigmoid=True)

    def forward(self, x):
        result = self.discriminator(x)
        result = result.view(-1, 1)
        return result


def get_discriminator(input_channels=3, final_sigmoid=True):
    discriminator_dict = OrderedDict([
            # block 1: 224x3 --> 112x64
            ("conv1", nn.Conv2d(input_channels, 64, 4, stride=2, padding=1)),
            ("batchnorm1", nn.BatchNorm2d(64, affine=False)),
            ("relu1", nn.LeakyReLU(negative_slope=0.2)),
            # block 2: 112x64 --> 56x128
            ("conv2", nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            ("batchnorm2", nn.BatchNorm2d(128, affine=False)),
            ("relu2", nn.LeakyReLU(negative_slope=0.2)),
            # block 3: 56x128 --> 28x256
            ("conv3", nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            ("batchnorm3", nn.BatchNorm2d(256, affine=False)),
            ("relu3", nn.LeakyReLU(negative_slope=0.2)),
            # block 4: 28x256 --> 14x512
            ("conv4", nn.Conv2d(256, 512, 4, stride=2, padding=1)),
            ("batchnorm4", nn.BatchNorm2d(512, affine=False)),
            ("relu4", nn.LeakyReLU(negative_slope=0.2)),
            # block 5: 14x512 --> 1x1
            ("conv5", nn.Conv2d(512, 1, 14, stride=1, padding=0)),
    ])
    if final_sigmoid:
        discriminator_dict.update({"sigmoid": nn.Sigmoid()})
    discriminator = nn.Sequential(discriminator_dict)
    return discriminator