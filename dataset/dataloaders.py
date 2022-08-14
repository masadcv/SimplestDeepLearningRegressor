import torch
import torchvision.transforms as transforms

from .mnist import MNIST

SUPPORTED_DATASETS = ["MNIST"]

data_to_train_transforms = {
    "MNIST": transforms.Compose(
        [
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            # normalise with mean/std
            transforms.Normalize((0.1307,), (0.3081,)),
            # normalise to -1 to 1 range
            # transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
}

data_to_test_transforms = {
    "MNIST": transforms.Compose(
        [
            transforms.ToTensor(),
            # normalise with mean/std
            transforms.Normalize((0.1307,), (0.3081,)),
            # normalise to -1 to 1 range
            # transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
}


def get_loaders(
    dataset,
    batch_size,
    path_to_data,
    num_workers,
    train_val_split,
    num_sel_train_images=-1,
    num_sel_test_images=-1,
):
    # get training and validation data loader
    train_loader, val_loader = get_train_val_loader(
        dataset,
        batch_size,
        path_to_data,
        num_workers,
        train_val_split=train_val_split,
        num_sel_images=num_sel_train_images,
    )

    # get testing data loader
    test_loader = get_test_loader(
        dataset,
        batch_size,
        path_to_data,
        num_workers,
        num_sel_images=num_sel_test_images,
    )

    return train_loader, val_loader, test_loader


def get_train_val_loader(
    dataset,
    batch_size,
    path_to_data,
    num_workers,
    train_val_split,
    num_sel_images,
):

    # check if data is implemented
    assert dataset in SUPPORTED_DATASETS

    # get relevant dataset transforms for train/val
    train_transform = data_to_train_transforms[dataset]
    val_transform = data_to_test_transforms[dataset]

    # get train/val datasets
    if dataset == "MNIST":
        train_data = MNIST(
            root=path_to_data,
            set="train",
            train_val_split=train_val_split,
            download=True,
            transform=train_transform,
            num_sel_images=num_sel_images,
        )
        val_data = MNIST(
            root=path_to_data,
            set="val",
            train_val_split=train_val_split,
            download=True,
            transform=val_transform,
            num_sel_images=num_sel_images,
        )
    else:
        raise ValueError("Unrecognised dataset {}".format(dataset))

    # init dataloader objects, train-> shuffle, val-> no shuffle
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


def get_test_loader(dataset, batch_size, path_to_data, num_workers, num_sel_images):
    # check if data is implemented
    assert dataset in SUPPORTED_DATASETS

    # get relevant dataset transforms for test
    test_transform = data_to_test_transforms[dataset]

    # get test dataset
    if dataset == "MNIST":
        val_data = MNIST(
            root=path_to_data,
            set="test",
            download=True,
            transform=test_transform,
            num_sel_images=num_sel_images,
        )
    else:
        raise ValueError("Unrecognised dataset {}".format(dataset))

    # init data loader object, test-> no shuffle
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return val_loader
