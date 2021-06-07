import torch

import utils


def train(args, model, train_loader, optimizer, loss_func, epoch):
    losses = utils.AverageMeter()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args["device"]), target.to(args["device"])
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        losses.update(loss.item(), data.size(0))
        optimizer.step()
        if batch_idx % args["log_interval"] == 0 and batch_idx != 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * train_loader.batch_size,
                    len(train_loader) * train_loader.batch_size,
                    100.0 * batch_idx / len(train_loader),
                    losses.avg,
                )
            )

    if "tb_writer" in args.keys():
        args["tb_writer"].add_scalar(
            "train_%s_loss" % args["epoch_name"], losses.avg, epoch
        )


def evaluate(args, model, test_loader, loss_func, epoch):
    losses = utils.AverageMeter()
    model.eval()
    cacc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args["device"]), target.to(args["device"])
            output = model(data)
            test_loss = torch.sum(loss_func(output, target))
            losses.update(test_loss.item(), data.size(0))
            # mean square error for accuracy
            cacc += (torch.sum((target - output) ** 2) / target.numel()).item()

    accuracy = cacc / (len(test_loader) * test_loader.batch_size)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n".format(
            losses.avg, accuracy
        )
    )

    if "tb_writer" in args.keys():
        args["tb_writer"].add_scalar(
            "test_%s_loss" % args["epoch_name"], losses.avg, epoch
        )
        args["tb_writer"].add_scalar(
            "test_%s_acc" % args["epoch_name"], accuracy, epoch
        )

    return accuracy