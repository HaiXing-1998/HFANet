import torch.nn as nn
from Attention_Module import HFAB
from High_Frequency_Module import HighFrequencyModule


class Encoder(nn.Module):
    def __init__(self, input_channel, input_size):
        super(Encoder, self).__init__()
        bn_momentum = 0.1
        # pre_treat_layer
        self._pre_treat_1 = HighFrequencyModule(input_channel,
                                                mode='high_boost_filtering',
                                                smooth=True)
        self._pre_treat_2 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=1, stride=1)
        # layer_1
        self._layer_1 = nn.Sequential(
            HFAB(input_channel=64, input_size=input_size),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.PReLU(64),
            HFAB(input_channel=64, input_size=input_size),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.PReLU(64)
        )
        # skip_connection_1 & down_sample
        self._down_sample_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer_2
        self._layer_2 = nn.Sequential(
            HFAB(input_channel=64, input_size=input_size // 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.PReLU(128),
            HFAB(input_channel=128, input_size=input_size // 2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.PReLU(128)
        )
        # skip_connection_2 & down_sample
        self._down_sample_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer_3
        self._layer_3 = nn.Sequential(
            HFAB(input_channel=128, input_size=input_size // 4),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.PReLU(256),
            HFAB(input_channel=256, input_size=input_size // 4),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.PReLU(256)
        )
        # skip_connection_3 & down_sample
        self._down_sample_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer_4
        self._layer_4 = nn.Sequential(
            HFAB(input_channel=256, input_size=input_size // 8),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.PReLU(512),
            HFAB(input_channel=512, input_size=input_size // 8),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.PReLU(512)
        )
        # skip_connection_4 & down_sample
        self._down_sample_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer_5
        self._layer_5 = nn.Sequential(
            HFAB(input_channel=512, input_size=input_size // 16),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024, momentum=bn_momentum),
            nn.PReLU(1024),
            HFAB(input_channel=1024, input_size=input_size // 16),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024, momentum=bn_momentum),
            nn.PReLU(1024)
        )

    def forward(self, x):
        # pre-treat layer
        x = self._pre_treat_1(x)
        x = self._pre_treat_2(x)
        # layer 1
        x = self._layer_1(x)
        skip_1 = x
        x = self._down_sample_1(x)
        # layer 2
        x = self._layer_2(x)
        skip_2 = x
        x = self._down_sample_2(x)
        # layer 3
        x = self._layer_3(x)
        skip_3 = x
        x = self._down_sample_3(x)
        # layer 4
        x = self._layer_4(x)
        skip_4 = x
        x = self._down_sample_4(x)
        x = self._layer_5(x)
        return x, skip_1, skip_2, skip_3, skip_4

