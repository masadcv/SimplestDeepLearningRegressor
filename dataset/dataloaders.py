import torch
import torchvision.transforms as transforms

from .mnist import MNIST

SUPPORTED_DATASETS = ["MNIST"]

data_to_train_transforms = {
    "MNIST": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
}

data_to_valid_transforms = {
    "MNIST": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
}


def get_loaders(
    dataset,
    batch_size,
    path_to_data,
    num_workers,
    num_sel_train_images=-1,
    num_sel_test_images=-1,
):
    train_loader = get_train_loader(
        dataset,
        batch_size,
        path_to_data,
        num_workers,
        num_sel_images=num_sel_train_images,
    )
    val_loader = get_val_loader(
        dataset,
        batch_size,
        path_to_data,
        num_workers,
        num_sel_images=num_sel_test_images,
    )

    return train_loader, val_loader


def get_train_loader(dataset, batch_size, path_to_data, num_workers, num_sel_images):

    assert dataset in SUPPORTED_DATASETS

    train_transform = data_to_train_transforms[dataset]

    if dataset == "MNIST":
        train_data = MNIST(
            path_to_data, train=True, download=True, transform=train_transform
        )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader


def get_val_loader(dataset, batch_size, path_to_data, num_workers, num_sel_images):

    assert dataset in SUPPORTED_DATASETS

    val_transform = data_to_valid_transforms[dataset]

    if dataset == "MNIST":
        val_data = MNIST(path_to_data, train=False, transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return val_loader
