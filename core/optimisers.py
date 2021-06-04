import torch

# TODO: add more parameters such as momementum etc
def get_optimiser(optimiser_name, params, lr):
    optimiser_dict = {
        "SGD": torch.optim.SGD,
        "ADAM": torch.optim.Adam,
        "RMSProp": torch.optim.RMSprop,
    }

    return optimiser_dict[optimiser_name](params, lr=lr)