import torch.nn as nn
import torch

from Attention_Module import HFAB


# from High_Frequency_Module import HighFrequencyModule


class Decoder(nn.Module):
    def __init__(self, input_channel, input_size):
        super(Decoder, self).__init__()
        bn_momentum = 0.1
        # up_sample_1
        # self._up_sample_1 = nn.ConvTranspose2d(input_channel, input_channel // 2,
        #                                       kernel_size=2,
        #                                       stride=2)
        self._up_sample_1 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel // 2, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # up_layer_1
        self._up_layer_1 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 2, momentum=bn_momentum),
            nn.PReLU(input_channel // 2),
            HFAB(input_channel=input_channel // 2, input_size=input_size * 2),
            nn.Conv2d(input_channel // 2, input_channel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 2, momentum=bn_momentum),
            nn.PReLU(input_channel // 2),
            HFAB(input_channel=input_channel // 2, input_size=input_size * 2)
        )
        # up_sample_2
        # self._up_sample_2 = nn.ConvTranspose2d(input_channel // 2, input_channel // 4,
        #                                        kernel_size=2,
        #                                        stride=2)
        self._up_sample_2 = nn.Sequential(
            nn.Conv2d(input_channel // 2, input_channel // 4, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # up_layer_2
        self._up_layer_2 = nn.Sequential(
            nn.Conv2d(input_channel // 2, input_channel // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 4, momentum=bn_momentum),
            nn.PReLU(input_channel // 4),
            HFAB(input_channel=input_channel // 4, input_size=input_size * 4),
            nn.Conv2d(input_channel // 4, input_channel // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 4, momentum=bn_momentum),
            nn.PReLU(input_channel // 4),
            HFAB(input_channel=input_channel // 4, input_size=input_size * 4)
        )
        # up_sample_3
        # self._up_sample_3 = nn.ConvTranspose2d(input_channel // 4, input_channel // 8,
        #                                        kernel_size=2,
        #                                        stride=2)
        self._up_sample_3 = nn.Sequential(
            nn.Conv2d(input_channel // 4, input_channel // 8, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # up_layer_3
        self._up_layer_3 = nn.Sequential(
            nn.Conv2d(input_channel // 4, input_channel // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 8, momentum=bn_momentum),
            nn.PReLU(input_channel // 8),
            HFAB(input_channel=input_channel // 8, input_size=input_size * 8),
            nn.Conv2d(input_channel // 8, input_channel // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 8, momentum=bn_momentum),
            nn.PReLU(input_channel // 8),
            HFAB(input_channel=input_channel // 8, input_size=input_size * 8)
        )
        # up_sample_4
        #self._up_sample_4 = nn.ConvTranspose2d(input_channel // 8, input_channel // 16,
        #                                       kernel_size=2,
        #                                       stride=2)
        self._up_sample_4 = nn.Sequential(
            nn.Conv2d(input_channel // 8, input_channel // 16, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # up_layer_4
        self._up_layer_4 = nn.Sequential(
            nn.Conv2d(input_channel // 8, input_channel // 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 16, momentum=bn_momentum),
            nn.PReLU(input_channel // 16),
            HFAB(input_channel=input_channel // 16, input_size=input_size * 16),
            nn.Conv2d(input_channel // 16, input_channel // 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 16, momentum=bn_momentum),
            nn.PReLU(input_channel // 16),
            HFAB(input_channel=input_channel // 16, input_size=input_size * 16)
        )
        # out_layer
        self._out_layer = nn.Sequential(
            nn.Conv2d(input_channel // 16, input_channel // 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 32, momentum=bn_momentum),
            nn.PReLU(input_channel // 32),
            nn.Conv2d(input_channel // 32, input_channel // 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 32, momentum=bn_momentum),
            nn.PReLU(input_channel // 32),
            nn.Conv2d(input_channel // 32, 1, kernel_size=1, stride=1)
        )

    def forward(self, x, skip_1, skip_2, skip_3, skip_4):
        # up layer 1 & concat skip connection -1
        x = self._up_sample_1(x)
        x = torch.cat((x, skip_4), dim=1)
        x = self._up_layer_1(x)
        # up layer 2 & concat skip connection -2
        x = self._up_sample_2(x)
        x = torch.cat((x, skip_3), dim=1)
        x = self._up_layer_2(x)
        # up layer 3 & concat skip connection -3
        x = self._up_sample_3(x)
        x = torch.cat((x, skip_2), dim=1)
        x = self._up_layer_3(x)
        # up layer 4 & concat skip connection -4
        x = self._up_sample_4(x)
        x = torch.cat((x, skip_1), dim=1)
        x = self._up_layer_4(x)
        # out layer
        x = self._out_layer(x)
        return x
