import torch.nn as nn
from torch.nn.modules import loss

class CombinedMSEMAE(nn.Module):
    def __init__(self):
        super(CombinedMSEMAE, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, output, target):
        comb = 0.5 * self.mse(output, target) + 0.5 * self.mae(output, target)
        return comb


def get_loss(name):
    lossname_to_func = {
        'XENT': nn.CrossEntropyLoss(),
        'MSE': nn.MSELoss(),
        'MAE': nn.L1Loss(),
        'SMOOTHMAE': nn.SmoothL1Loss(),

        # our custom loss
        'COMB': CombinedMSEMAE(),
    }
    return lossname_to_func[name]