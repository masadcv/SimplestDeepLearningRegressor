import argparse
import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter

import core
import dataset
import models
import utils


def parse_args():
    parser = argparse.ArgumentParser("Neural Network Trainer")
    parser.add_argument("--model", type=str, default="mnistcnn", help="model to learn")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_sl1.json",
        help="training config parameters",
        required=False,
    )
    args = parser.parse_args()
    return args


def main():
    inargs = parse_args()

    # load config
    exp_config = utils.load_json(inargs.config)

    exp_config["model"] = inargs.model

    # create working directories
    work_dir = os.path.join(exp_config["work_dir"], inargs.model)
    os.makedirs(work_dir, exist_ok=True)

    # run on cuda or cpu?
    use_cuda = exp_config["training"]["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataloaders
    dataloader_config = exp_config["dataloader"]
    train_loader, val_loader, test_loader = dataset.dataloaders.get_loaders(
        dataloader_config["dataset"],
        batch_size=dataloader_config["batch_size"],
        path_to_data=dataloader_config["path_to_data"],
        num_workers=dataloader_config["num_workers"],
        train_val_split=dataloader_config["train_val_split"],
        num_sel_train_images=dataloader_config["num_sel_train_images"],
        num_sel_test_images=dataloader_config["num_sel_test_images"],
    )

    model_to_func = models.model_zoo.modelname_to_func

    # check if model exists in the zoo
    model_list = list(model_to_func.keys())
    assert (
        inargs.model in model_list
    ), "Invalid model provided - can only be from [{}]".format(model_list)

    model = model_to_func[inargs.model](
        out_channels=dataloader_config["out_channels"],
        use_bn=exp_config["training"]["use_batchnorm"],
        activation=exp_config["training"]["activation"],
        dropout=exp_config["training"]["dropout"],
    ).to(device)
    print(model)

    # setup optimiser
    optim_config = exp_config["optimiser"]
    optimiser = core.optimisers.get_optimiser(
        optim_config["type"], params=model.parameters(), lr=optim_config["lr"]
    )
    # setup scheduler
    scheduler = core.schedulers.get_scheduler(
        type=optim_config["scheduler_type"],
        optim=optimiser,
        lr_step=optim_config["scheduler_lr_step"],
        lr_factor=optim_config["scheduler_lr_factor"],
        epochs=exp_config["training"]["epochs"],
    )

    # setup loss
    loss_config = exp_config["loss"]
    loss_func = core.loss.get_loss(loss_config["type"], reduction="mean")

    # create the tensorboard folder from date/time to log training params
    current_date = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M")
    tensorboard_folder = os.path.join(
        work_dir,
        current_date + "{}_{}".format(loss_config["type"], optim_config["type"]),
    )
    tb_writer = SummaryWriter(log_dir=tensorboard_folder)

    # save trainer config to training folder
    utils.save_json(exp_config, os.path.join(tensorboard_folder, "config.json"))

    # configure and run the training loop
    train_config = exp_config["training"]
    args = {}
    args["log_interval"] = train_config["log_interval"]
    args["device"] = device
    args["tb_writer"] = tb_writer

    metrics_dict = {}
    for vm in train_config["val_metrics"]:
        metrics_dict[vm] = core.metrics.get_metric(vm).to(device)
    args["val_metrics"] = metrics_dict
    best_loss_val = 1e10

    for epoch in range(0, train_config["epochs"]):

        # run training loop
        print("-" * 50)
        args["epoch_name"] = "train"
        core.function.train(args, model, train_loader, optimiser, loss_func, epoch)
        print("-" * 50)

        if epoch % train_config["eval_interval"] == 0:
            args["epoch_name"] = "val"
            accuracy, loss_val = core.function.evaluate(
                args, model, val_loader, loss_func, epoch
            )

            # run test iteration, not used for early stopping or tuning
            if train_config["run_test_epoch"]:
                args["epoch_name"] = "test"
                _, _ = core.function.evaluate(
                    args, model, test_loader, loss_func, epoch
                )

            if loss_val < best_loss_val:
                best_loss_val = loss_val
                print(
                    "Found best validation loss {:.4f}, saving model...".format(
                        loss_val
                    )
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(tensorboard_folder, "learned_model.pt"),
                )

        scheduler.step()


if __name__ == "__main__":
    main()
