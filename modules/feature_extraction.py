import torch.nn as nn
import torch.nn.functional as F
import torch
from modules.rbn import RepresentativeBatchNorm2d
import modules.resnet as ResNet
# from modules.fpem_v1 import FPEM_v1
# from modules.fpem_v2 import FPEM_v2
# from modules.conv_bn_relu import Conv_BN_ReLU
"""
2.特征提取
"""
class ResNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel=1, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])


    def forward(self, input):
        output = self.ConvNet(input)
        return output