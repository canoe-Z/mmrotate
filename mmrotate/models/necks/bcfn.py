import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmengine.model import BaseModule, xavier_init

from mmrotate.registry import MODELS
from mmdet.utils import OptConfigType


class ChannelInteractionModule(BaseModule):
    def __init__(self,
                 feat_channels: int = 256,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super(ChannelInteractionModule, self).__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.init_cfg = init_cfg

        self.fc1 = ConvModule(
            feat_channels,
            feat_channels,
            1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        self.fc2 = ConvModule(
            feat_channels,
            feat_channels,
            1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)

    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                xavier_init(m, distribution='uniform')

    def forward(self, x1, x2):
        """Forward function."""
        x1_1, x1_2 = torch.chunk(x1, 2, dim=1)
        x2_1, x2_2 = torch.chunk(x1, 2, dim=1)
        x1 = torch.cat([x1_1, x2_2], dim=1)
        x2 = torch.cat([x2_1, x1_2], dim=1)
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)

        return x1, x2


@MODELS.register_module()
class BCFN(BaseModule):
    def __init__(self,
                 num_ins: int = 5,
                 feat_channels: int = 256,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super(BCFN, self).__init__(init_cfg)
        self.num_ins = num_ins
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.init_cfg = init_cfg

        self.cim = ChannelInteractionModule(feat_channels)
        self.pw_conv = ConvModule(
            feat_channels,
            feat_channels,
            1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        upsample_cfg = dict(mode='nearest')
        self.upsample_cfg = upsample_cfg.copy()
        self.gamma = nn.Parameter(torch.ones(
            (1, feat_channels, 1, 1)), requires_grad=True)

    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                xavier_init(m, distribution='uniform')
        self.cim.init_weights()

    def forward(self, inputs):
        """Forward function."""
        inputs = list(inputs)

        for i in range(len(inputs[:-1])):
            prev_shape = inputs[i].shape[2:]
            upsampled_feat = F.interpolate(
                inputs[i + 1], size=prev_shape, **self.upsample_cfg)

            feat1, feat2 = self.cim(inputs[i], upsampled_feat)
            feat = feat1 * feat2
            feat = self.pw_conv(feat)

            inputs[i] = inputs[i] + self.gamma * feat

        return inputs[:-1]
