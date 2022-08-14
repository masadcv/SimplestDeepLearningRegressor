import torch


def get_optimiser(optimiser_name, params, lr):
    optimiser_dict = {
        "SGD": torch.optim.SGD,
        "ADAM": torch.optim.Adam,
        "ADADELTA": torch.optim.Adadelta,
        "RMSProp": torch.optim.RMSprop,
    }

    return optimiser_dict[optimiser_name](params, lr=lr)
