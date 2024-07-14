import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer
from mmdet.registry import MODELS


@MODELS.register_module()
class MyBackbone(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), **kwargs):
        super(MyBackbone, self).__init__()
        self.conv1 = build_conv_layer(None, in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = build_norm_layer(norm_cfg, out_channels)[1]
        self.activation = build_activation_layer(act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x

    def init_weights(self):
        # 使用 mmengine 提供的初始化函数
        from mmengine.model import constant_init, normal_init
        normal_init(self.conv1, mean=0, std=0.01)
        constant_init(self.norm1, 1)