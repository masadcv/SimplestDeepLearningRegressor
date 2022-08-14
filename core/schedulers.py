import torch


def get_scheduler(type, optim, lr_step, lr_factor, epochs):
    if type == "MULTI":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim,
            [int(epochs * lrstep) for lrstep in list(lr_step)],
            gamma=lr_factor,
        )
    elif type == "COSINE":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    return scheduler
