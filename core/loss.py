import torch.nn as nn


def get_loss(name, reduction="mean"):
    lossname_to_func = {
        "MSE": nn.MSELoss(reduction=reduction),
        "MAE": nn.L1Loss(reduction=reduction),
        "SMOOTHMAE": nn.SmoothL1Loss(beta=0.01, reduction=reduction),
    }
    return lossname_to_func[name]
