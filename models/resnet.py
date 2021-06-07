from collections import OrderedDict
from typing import Any, List, Type, Union

import torch.nn as nn
from torch.functional import Tensor
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, model_urls


class ResNetMNIST(ResNet):
    def __init__(self, block, layers, **kwargs):
        super(ResNetMNIST, self).__init__(block, layers, **kwargs)

        self.num_classes = kwargs.get("num_classes", 1000)

        # our input is one channel, imagenet input was 3 channel
        # so need to delete and reinit first layer
        del self.conv1

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def load_state_dict(self, state_dict: "OrderedDict[str, Tensor]", strict: bool):
        state_dict.pop("conv1.weight")
        # state_dict.pop("conv1.bias")

        if self.num_classes != 1000:
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")

        return super().load_state_dict(state_dict, strict=strict)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNetMNIST:
    model = ResNetMNIST(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNetMNIST:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNetMNIST:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)
