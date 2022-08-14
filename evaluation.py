import argparse
import os

import torch

import core
import dataset
import models
import utils


def parse_args():
    parser = argparse.ArgumentParser("Neural Network Evaluator")
    parser.add_argument(
        "--folder",
        type=str,
        default="experiments/folder/to",
        help="folder with trained model and config",
        required=False,
    )
    args = parser.parse_args()
    return args


def main():
    inargs = parse_args()

    config_file = os.path.join(inargs.folder, "config.json")
    model_file = os.path.join(inargs.folder, "learned_model.pt")

    if not os.path.exists(config_file):
        raise IOError("Unable to find config file: {}".format(config_file))

    if not os.path.exists(model_file):
        raise IOError("Unable to find model file: {}".format(model_file))

    # load config
    exp_config = utils.load_json(config_file)
    model_name = exp_config["model"]

    # run on cuda or cpu?
    use_cuda = exp_config["training"]["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataloaders
    dataloader_config = exp_config["dataloader"]
    _, _, test_loader = dataset.dataloaders.get_loaders(
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
        model_name in model_list
    ), "Invalid model provided - can only be from [{}]".format(model_list)

    model = model_to_func[model_name](
        out_channels=dataloader_config["out_channels"],
        use_bn=exp_config["training"]["use_batchnorm"],
        activation=exp_config["training"]["activation"],
        dropout=exp_config["training"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_file))
    print(model)

    # setup loss
    loss_config = exp_config["loss"]
    loss_func = core.loss.get_loss(loss_config["type"])

    # configure and run the training loop
    train_config = exp_config["training"]
    fargs = {}
    fargs["log_interval"] = train_config["log_interval"]
    fargs["device"] = device

    metrics_dict = {}
    for vm in train_config["val_metrics"]:
        metrics_dict[vm] = core.metrics.get_metric(vm).to(device)

    fargs["val_metrics"] = metrics_dict

    # run evaluation loop
    (
        accuracy,
        loss_val,
        out_pred,
        out_target,
    ) = core.function.evaluate_with_results(fargs, model, test_loader, loss_func, 0)

    # conf_matrix_func_res = core.metrics.get_metric(
    #     "CONF", num_classes=10
    # )
    # utils.make_confusion_matrix_plot(
    #     out_pred,
    #     out_target,
    #     conf_matrix_func_res,
    #     save_path=os.path.join(inargs.folder, "conf_res.png"),
    # )
    utils.save_json(accuracy, os.path.join(inargs.folder, "test_accuracy.json"))


if __name__ == "__main__":
    main()
