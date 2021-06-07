import argparse
import datetime
import os

import tensorboardX
import torch

import core.function
import core.loss
import core.optimisers
import dataset.dataloaders
import models.model_zoo
import utils


def parse_args():
    parser = argparse.ArgumentParser("Basic Neural Network Trainer")
    parser.add_argument("--model", type=str, default="resnet18", help="model to learn")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json",
        help="training config parameters",
        required=False,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    exp_config = utils.load_json(args.config)

    # create working directories
    work_dir = os.path.join(exp_config["work_dir"], args.model)
    os.makedirs(work_dir, exist_ok=True)

    # run on cuda or cpu?
    use_cuda = exp_config["training"]["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataloaders
    dataloader_config = exp_config["dataloader"]
    train_loader, val_loader = dataset.dataloaders.get_loaders(
        dataloader_config["dataset"],
        batch_size=dataloader_config["batch_size"],
        path_to_data=dataloader_config["path_to_data"],
        num_workers=dataloader_config["num_workers"],
        num_sel_train_images=dataloader_config["num_sel_train_images"],
        num_sel_test_images=dataloader_config["num_sel_test_images"],
    )

    # create the tensorboard folder from date/time to log training params
    current_date = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M")
    tensorboard_folder = os.path.join(work_dir, current_date)
    tb_writer = tensorboardX.SummaryWriter(log_dir=tensorboard_folder)

    model_to_func = models.model_zoo.modelname_to_func

    # check if model exists in the zoo
    model_list = list(model_to_func.keys())
    assert (
        args.model in model_list
    ), "Invalid model provided - can only be from [%s]" % (model_list)

    model = model_to_func[args.model](
        pretrained=False, num_classes=exp_config["dataloader"]["num_classes"]
    ).to(device)

    # setup optimiser
    optim_config = exp_config["optimiser"]
    optimiser = core.optimisers.get_optimiser(
        optim_config["type"], params=model.parameters(), lr=optim_config["lr"]
    )

    # setup loss
    loss_config = exp_config["loss"]
    loss_func = core.loss.get_loss(loss_config["type"])

    # configure and run the training loop
    train_config = exp_config["training"]
    args = {}
    args["log_interval"] = train_config["log_interval"]
    args["device"] = device
    args["tb_writer"] = tb_writer

    # TODO: add custom metric

    for epoch in range(0, train_config["epochs"]):
        print("-" * 50)
        args["epoch_name"] = "train"
        core.function.train(args, model, train_loader, optimiser, loss_func, epoch)
        print("-" * 50)

        if epoch % train_config["eval_interval"] == 0:
            args["epoch_name"] = ""
            core.function.evaluate(args, model, val_loader, loss_func, epoch)

        # TODO: save modle that has good validation

    torch.save(model.state_dict(), os.path.join(work_dir, "learned_model.pt"))
    return model


if __name__ == "__main__":
    main()
