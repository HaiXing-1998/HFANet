import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder


class HFANet(nn.Module):
    def __init__(self, input_channel, input_size):
        super(HFANet, self).__init__()
        self.encoder = Encoder(input_channel=input_channel, input_size=input_size)
        self.decoder = Decoder(input_channel=1024, input_size=16)
        self.skip_connection_feature_fusion_1 = nn.Conv2d(64 * 2, 64,
                                                          kernel_size=1,
                                                          stride=1)
        self.skip_connection_feature_fusion_2 = nn.Conv2d(128 * 2, 128,
                                                          kernel_size=1,
                                                          stride=1)
        self.skip_connection_feature_fusion_3 = nn.Conv2d(256 * 2, 256,
                                                          kernel_size=1,
                                                          stride=1)
        self.skip_connection_feature_fusion_4 = nn.Conv2d(512 * 2, 512,
                                                          kernel_size=1,
                                                          stride=1)
        self.bottom_feature_fusion = nn.Conv2d(1024 * 2, 1024,
                                               kernel_size=1,
                                               stride=1)

    def forward(self, t1, t2):
        bottom_feature_1, skip_connect_1_1, skip_connect_1_2, skip_connect_1_3, skip_connect_1_4 = self.encoder(
            t1)
        bottom_feature_2, skip_connect_2_1, skip_connect_2_2, skip_connect_2_3, skip_connect_2_4 = self.encoder(
            t2)
        skip_connect_fusion_1 = torch.cat((skip_connect_1_1, skip_connect_2_1), dim=1)
        skip_connect_fusion_2 = torch.cat((skip_connect_1_2, skip_connect_2_2), dim=1)
        skip_connect_fusion_3 = torch.cat((skip_connect_1_3, skip_connect_2_3), dim=1)
        skip_connect_fusion_4 = torch.cat((skip_connect_1_4, skip_connect_2_4), dim=1)
        bottom_fusion = torch.cat((bottom_feature_1, bottom_feature_2), dim=1)
        skip_connect_final_1 = self.skip_connection_feature_fusion_1(skip_connect_fusion_1)
        skip_connect_final_2 = self.skip_connection_feature_fusion_2(skip_connect_fusion_2)
        skip_connect_final_3 = self.skip_connection_feature_fusion_3(skip_connect_fusion_3)
        skip_connect_final_4 = self.skip_connection_feature_fusion_4(skip_connect_fusion_4)
        bottom_final = self.bottom_feature_fusion(bottom_fusion)
        output = self.decoder(bottom_final,
                              skip_connect_final_1,
                              skip_connect_final_2,
                              skip_connect_final_3,
                              skip_connect_final_4)
        return output
