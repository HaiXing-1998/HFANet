import torch
import torch.nn as nn
from High_Frequency_Module import HighFrequencyModule

class HighFrequencyEnhancementStage(nn.Module):
    def __init__(self, input_channel, input_size, ratio=0.5):
        super(HighFrequencyEnhancementStage, self).__init__()
        self.input_channel = input_channel
        self.input_size = input_size
        self.ratio_channel = int(ratio * input_channel)
        self.Global_pooling = nn.AvgPool2d(self.input_size)
        self.FC_1 = nn.Linear(self.input_channel, int(self.input_channel * ratio))
        self.ReLU = nn.PReLU(int(self.input_channel * ratio))
        self.FC_2 = nn.Linear(int(self.input_channel * ratio), self.input_channel)
        self.Sigmoid = nn.Sigmoid()
        self.HighFre = HighFrequencyModule(input_channel=self.input_channel,smooth=True)
        self.Channelfusion = nn.Conv2d(2 * self.input_channel, self.input_channel, kernel_size=1, stride=1)

    # ChannelAttention +HighFrequency
    def forward(self, x):
        residual = x  # residual & x's shape [batch size, channel, input size, input size]
        x_hf = self.HighFre(residual)
        x = self.Global_pooling(x)  # x's shape [batch size, channel, 1, 1]
        x = x.view(-1, self.input_channel)  # x's shape [batch size, channel]
        x = self.FC_1(x)  # x's shape [batch size, ratio channel]
        x = self.ReLU(x)
        x = self.FC_2(x)  # x's shape [batch size, channel]
        x = self.Sigmoid(x)
        x = torch.unsqueeze(x, dim=2)  # x's shape [batch size, channel, 1]
        residual_0 = residual.view(-1, self.input_channel, self.input_size ** 2)
        residual_0 = torch.mul(residual_0, x)
        residual_0 = residual_0.contiguous().view(-1, self.input_channel, self.input_size, self.input_size)
        x_output = residual + residual_0
        x_output = torch.cat((x_output, x_hf), dim=1)
        x_output = self.Channelfusion(x_output)
        return x_output


class SpatialAttentionStage(nn.Module):
    def __init__(self, input_channel):
        super(SpatialAttentionStage, self).__init__()
        self.bn_momentum = 0.1
        self.input_channel = input_channel
        # down 1
        self.conv1_1 = nn.Conv2d(self.input_channel, self.input_channel // 2,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn1_1 = nn.BatchNorm2d(self.input_channel // 2,
                                    momentum=self.bn_momentum)
        self.ReLU1_1 = nn.PReLU(self.input_channel // 2)
        self.conv1_2 = nn.Conv2d(self.input_channel // 2, self.input_channel // 2,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn1_2 = nn.BatchNorm2d(self.input_channel // 2,
                                    momentum=self.bn_momentum)
        self.ReLU1_2 = nn.PReLU(self.input_channel // 2)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2,
                                        stride=2)
        # down 2
        self.conv2_1 = nn.Conv2d(self.input_channel // 2, self.input_channel // 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn2_1 = nn.BatchNorm2d(self.input_channel // 4,
                                    momentum=self.bn_momentum)
        self.ReLU2_1 = nn.PReLU(self.input_channel // 4)
        self.conv2_2 = nn.Conv2d(self.input_channel // 4, self.input_channel // 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn2_2 = nn.BatchNorm2d(self.input_channel // 4,
                                    momentum=self.bn_momentum)
        self.ReLU2_2 = nn.PReLU(self.input_channel // 4)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2,
                                        stride=2)
        # bottom
        self.conv_b_1 = nn.Conv2d(self.input_channel // 4, self.input_channel // 8,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.bn_b_1 = nn.BatchNorm2d(self.input_channel // 8,
                                     momentum=self.bn_momentum)
        self.ReLU_b_1 = nn.PReLU(self.input_channel // 8)
        self.conv_b_2 = nn.Conv2d(self.input_channel // 8, self.input_channel // 8,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.bn_b_2 = nn.BatchNorm2d(self.input_channel // 8,
                                     momentum=self.bn_momentum)
        self.ReLU_b_2 = nn.PReLU(self.input_channel // 8)
        # up 1
        self.convtrans_1 = nn.ConvTranspose2d(self.input_channel // 8, self.input_channel // 16,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              output_padding=1)
        self.conv3_1 = nn.Conv2d(self.input_channel // 16 + self.input_channel // 4, self.input_channel // 16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn3_1 = nn.BatchNorm2d(self.input_channel // 16,
                                    momentum=self.bn_momentum)
        self.ReLU3_1 = nn.PReLU(self.input_channel // 16)
        self.conv3_2 = nn.Conv2d(self.input_channel // 16, self.input_channel // 16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn3_2 = nn.BatchNorm2d(self.input_channel // 16,
                                    momentum=self.bn_momentum)
        self.ReLU3_2 = nn.PReLU(self.input_channel // 16)
        # up 2
        self.convtrans_2 = nn.ConvTranspose2d(self.input_channel // 16, self.input_channel // 32,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              output_padding=1)
        self.conv4_1 = nn.Conv2d(self.input_channel // 32 + self.input_channel // 2, self.input_channel // 32,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn4_1 = nn.BatchNorm2d(self.input_channel // 32,
                                    momentum=self.bn_momentum)
        self.ReLU4_1 = nn.PReLU(self.input_channel // 32)
        self.conv4_2 = nn.Conv2d(self.input_channel // 32, self.input_channel // 32,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn4_2 = nn.BatchNorm2d(self.input_channel // 32,
                                    momentum=self.bn_momentum)
        self.ReLU4_2 = nn.PReLU(self.input_channel // 32)
        # out
        self.conv5_1 = nn.Conv2d(self.input_channel // 32, self.input_channel // 64,
                                 kernel_size=1,
                                 stride=1)
        self.bn5_1 = nn.BatchNorm2d(self.input_channel // 64,
                                    momentum=self.bn_momentum)
        self.ReLU5_1 = nn.PReLU(self.input_channel // 64)
        self.conv5_2 = nn.Conv2d(self.input_channel // 64, 1,
                                 kernel_size=1,
                                 stride=1)
        self.bn5_2 = nn.BatchNorm2d(1,
                                    momentum=self.bn_momentum)
        # self.ReLU5_2 = nn.PReLU(1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        # down 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.ReLU1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.ReLU1_2(x)
        # skip connection
        skip_1 = x
        x = self.maxpooling1(x)
        # down 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.ReLU2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.ReLU2_2(x)
        # skip connection
        skip_2 = x
        x = self.maxpooling2(x)
        # bottom
        x = self.conv_b_1(x)
        x = self.bn_b_1(x)
        x = self.ReLU_b_1(x)
        x = self.conv_b_2(x)
        x = self.bn_b_2(x)
        x = self.ReLU_b_2(x)
        # up 1
        x = self.convtrans_1(x)
        # cat skip connection
        x = torch.cat((x, skip_2), dim=1)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.ReLU3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.ReLU3_2(x)
        # up 2
        x = self.convtrans_2(x)
        # cat skip connection
        x = torch.cat((x, skip_1), dim=1)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.ReLU4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.ReLU4_2(x)
        # out
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.ReLU5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.Sigmoid(x)
        mask = torch.mul(residual, x)
        output = residual + mask
        return output


class HFAB(nn.Module):
    def __init__(self, input_channel, input_size, ratio=0.5):
        super(HFAB, self).__init__()
        self.SA = SpatialAttentionStage(input_channel=input_channel)
        self.HF = HighFrequencyEnhancementStage(input_channel=input_channel,
                                                               input_size=input_size,
                                                               ratio=ratio)
    def forward(self, x):
        x = self.SA(x)
        x = self.HF(x)

        return x

