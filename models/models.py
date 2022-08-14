import torch
import torch.nn as nn

from models.layers import Conv2dBNAct, LinearBNAct, init_weights
import torch.nn.functional as F


class MNistMLP(nn.Module):
    def __init__(
        self,
        in_channels=28 * 28,
        out_channels=1,
        mlp_layers=[32, 64],
        head_layers=[128],
        use_bn=True,
        activation="relu",
        dropout=0.3,
    ):
        super().__init__()

        # add fully-connected layers to get featuremaps from our mlp network
        self.mlpnetwork = []
        in_channels_current_layer = in_channels
        for hlayer in mlp_layers:
            self.mlpnetwork.append(
                LinearBNAct(
                    in_features=in_channels_current_layer,
                    out_features=hlayer,
                    use_bias=True,
                    use_bn=use_bn,
                    activation=activation,
                    dropout=dropout,
                )
            )
            in_channels_current_layer = hlayer

        # convert all collection of mlp layers to nn.Squential module
        self.mlpnetwork = nn.Sequential(*self.mlpnetwork)

        # add output head, to infer the number
        self.head = []
        for hdlayer in head_layers:
            self.head.append(
                LinearBNAct(
                    in_features=in_channels_current_layer,
                    out_features=hdlayer,
                    use_bias=True,
                    use_bn=use_bn,
                    activation=activation,
                    dropout=dropout,
                )
            )
            in_channels_current_layer = hdlayer

        # last layer in head: a linear mapping without any activation/bias or batchnorm
        self.head.append(
            LinearBNAct(
                in_features=in_channels_current_layer,
                out_features=out_channels,
                use_bias=True,
                use_bn=False,
                activation=False,
                dropout=None,
            )
        )
        # convert all collection of head layers to nn.Squential module
        self.head = nn.Sequential(*self.head)

        self.mlpnetwork.apply(init_weights)
        self.head.apply(init_weights)

    def forward(self, x):
        # flatten input to use in mlp layers
        x = torch.flatten(x, start_dim=1)

        # apply mlpnetwork to get features
        xf = self.mlpnetwork(x)

        # apply head to get output
        xout = self.head(xf)

        return xout


class MNistCNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        conv_layers=[32, 64],
        head_layers=[128],
        use_bn=True,
        activation="relu",
        dropout=0.3,
    ):
        super().__init__()

        # add convolution layers to get featuremaps
        self.convnetwork = []
        in_channels_current_layer = in_channels
        for clayer in conv_layers:
            self.convnetwork.append(
                Conv2dBNAct(
                    in_channels=in_channels_current_layer,
                    out_channels=clayer,
                    kernel_size=3,
                    stride=1,
                    use_bn=use_bn,
                    activation=activation,
                )
            )
            in_channels_current_layer = clayer

        # convert all collection of mlp layers to nn.Squential module
        self.convnetwork = nn.Sequential(*self.convnetwork)

        # add output head, to infer the number
        self.head = []
        in_channels_current_layer = 14 * 14 * 64
        for hdlayer in head_layers:
            self.head.append(
                LinearBNAct(
                    in_features=in_channels_current_layer,
                    out_features=hdlayer,
                    use_bn=use_bn,
                    activation=activation,
                    dropout=dropout,
                )
            )

            in_channels_current_layer = hdlayer

        # last layer in head: a linear mapping without any activation/bias or batchnorm
        self.head.append(
            LinearBNAct(
                in_features=in_channels_current_layer,
                out_features=out_channels,
                use_bias=True,
                use_bn=False,
                activation=False,
                dropout=None,
            )
        )
        # convert all collection of head layers to nn.Squential module
        self.head = nn.Sequential(*self.head)

        self.convnetwork.apply(init_weights)
        self.head.apply(init_weights)

    def forward(self, x):
        # apply convnetwork to get features
        xf = self.convnetwork(x)
        xf = F.max_pool2d(xf, 2)

        # flatten
        xf = torch.flatten(xf, start_dim=1)

        # apply head to get output
        xout = self.head(xf)

        return xout


if __name__ == "__main__":
    print(MNistMLP())
    print(MNistCNN())
