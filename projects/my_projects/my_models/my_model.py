from mmdet.registry import MODELS
from mmdet.utils import ConfigType
import torch.nn as nn

@MODELS.register_module()
class MyModel(nn.Module):
    def __init__(self, backbone: ConfigType) -> None:
        super(MyModel, self).__init__()
        self.backbone = MODELS.build(backbone)

    def forward(self, x):
        return self.backbone(x)