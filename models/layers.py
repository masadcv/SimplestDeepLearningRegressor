import torch.nn as nn


def get_activation_from_name(name):
    name_to_act_funct = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }
    return name_to_act_funct[name]


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.normal_(m.weight)
        # some layers may not have bias, so skip if this isnt found
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # some layers may not have bias, so skip if this isnt found
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Conv2dBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding="same",
        use_bn=True,
        activation="relu",
    ):

        modules = []
        modules.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        if use_bn:
            modules.append(nn.BatchNorm2d(num_features=out_channels))
        if activation:
            modules.append(get_activation_from_name(activation)())

        super().__init__(*modules)


class LinearBNAct(nn.Sequential):
    def __init__(
        self,
        in_features,
        out_features,
        use_bias=True,
        use_bn=True,
        activation="relu",
        dropout=0.2,
    ):

        modules = []
        modules.append(
            nn.Linear(in_features=in_features, out_features=out_features, bias=use_bias)
        )
        if use_bn:
            modules.append(nn.BatchNorm1d(num_features=out_features))
        if activation:
            modules.append(get_activation_from_name(activation)())
        if dropout:
            modules.append(nn.Dropout(p=dropout))

        super().__init__(*modules)
